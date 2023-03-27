import matplotlib.pyplot as plt
import numpy as np
import statistics
import math
import pandas as pd
import pyodbc
import datetime as dt
import ig_modules_plot as igp
import glob
import os


class AccessDB:

    def __init__(self, path):
        self.path = path
        self.driver = [x for x in pyodbc.drivers() if x.startswith('Microsoft Access Driver')][0]
        self.conn_str = f"DRIVER={self.driver};" f"DBQ={self.path}"
        self.cnxn = pyodbc.connect(self.conn_str)
        self.crsr = self.cnxn.cursor()

    def return_crsr(self):
        return self.crsr

    def return_cnxn(self):
        return self.cnxn

    def tables(self):
        tables = []
        for table_info in self.crsr.tables(tableType='TABLE'):
            tables.append(table_info.table_name)
        return tables


class Oma:

    def __init__(self,path, MM, columns=[], dt_conv=False):
        if isinstance(MM, int):
            if MM < 10:
                MM = str(MM)
                MM = f"MM0{MM}"
            else:
                MM = str(MM)
                MM = f"MM{MM}"
        else:
            if len(MM) < 4:
                if len(MM) == 1:
                    MM = f"MM0{MM}"
                elif len(MM) == 2:
                    MM = f"MM{MM}"
                else:
                    raise TypeError("no valid input")
            elif len(MM) == 4:
                pass
            else:
                raise TypeError("no valid input")
        self.MM = MM
        self.columns = columns
        self.cnxn = AccessDB(path=path).return_cnxn()
        self.dt = dt_conv

    def return_dataframe(self):
        sql =f"SELECT * FROM {self.MM}"
        df = pd.read_sql(sql, self.cnxn)
        DCol = self.dcol()
        df = df.drop(DCol, axis=1)

        if self.dt is True and len(df) > 0:
            df["Datum"] = df["Datum"].dt.strftime("%Y/%m/%d")
            df["DateTime"] = df["Datum"] + " " + df["Uhrzeit"]
            df["DateTime"] = pd.to_datetime(df["DateTime"])
            df["DateTime"] = df["DateTime"].apply(lambda r: dt.datetime.strftime(r, '%Y-%m-%d %H:%M:%S'))
            df["DateTime"] = pd.to_datetime(df["DateTime"])
        else:
            pass

        df["Nummer"] = df["Nummer"].astype("int32")
        df["Futter"] = df["Futter"].astype("int8")
        df["Batch"] = df["Batch"].astype("int16")
        df["all_res_ok"] = df["all_res_ok"].astype("int8")
        df["ampule_present"] = df["ampule_present"].astype("int8")
        df["long_bubble_defect_check_OK"] = df["long_bubble_defect_check_OK"].astype("int8")
        df["dot_defect_check_OK"] = df["dot_defect_check_OK"].astype("int8")
        df['BodenBlaseOk'] = df['BodenBlaseOk'].astype("int8")

        return df

    def show_columns(self):
        df = self.return_dataframe()
        print(df.columns)

    def dcol(self):
        DCol = ['contour_ok', 'bottom_ok', 'pos_ok', 'rippl_cnt_s',
                'rippl_ampl_s', 'delta_d_s', 'rippl_cnt_b', 'rippl_ampl_b', 'delta_d_b',
                'rippl_cnt_h', 'rippl_ampl_h', 'H', 'H1', 'H2', 'DK', 'WS1', 'WS2',
                'LDS', 'DS', 'LVS', 'D', 'l', 'LVB', 'db', 'LDB', 'b1', 'B2', 'E_S',
                'E_B', 'E_DS', 'E_DB', 'E_H', 'KHS', 'KHB', 'KHH', 'KP', 'KR', 'rBA',
                'LB', 'SIGMA_R', 'BH', 'BE', 'E_BE', 'rBE', 'WBE', 'DKS', 'E_DKS', 'WK',
                'DKS2', 'DKSV', 'long_bubble_defect_cnt', 'dot_defect_cnt', 'LSI',
                'LBI', 'AVS', 'AVB', 'WD', 'BodenBlaseVol',
                'NennDurchmesser', 'NennLänge', 'RohrKlasse', 'FarbCode',
                'KartonWechsel', 'Reserve1', 'Reserve2', 'Reserve3', 'Reserve4', 'Reserve5']
        if len(self.columns) == 0:
            pass
        else:
            for a in self.columns:
                DCol.remove(a)
        return DCol


class TestData:

    def __init__(self, path, batches, table=[]):
        self.batchlist = batches
        self.cnxn = AccessDB(path=path).return_cnxn()
        self.table = table

    def return_db(self):
        if len(self.table) == 0:
            table = self.standard_tables()
        else:
            table = self.table
        sql = f"SELECT * FROM {table}"
        df = pd.read_sql(sql, self.cnxn)
        batchlist = self.batchlist.astype("str")
        df = df[df["Batch"].isin(batchlist)]

        return df

    @staticmethod
    def standard_tables(self):
        tables = ['Festigkeit', 'FTest', 'GTest', 'KTest', 'MTest', 'Ofen', 'Rahmenfüllanlage', 'Schweissanlage']
        return tables


def a_b_test(dataframe):
    """
    prüft im Falle eines vorhandenen Nachtests, welcher der Freigabetests ist (kleinerer Tl)

    @param dataframe: Testdata -> Festigkeit
    @type dataframe: pandas dataframe
    @return: Testdata -> Festigkeit nur mit freigabe Tests
    @rtype: pandas dataframe
    """
    fest = []
    batches = dataframe["Batch"].drop_duplicates()
    for a in batches:
        df = dataframe[dataframe["Batch"] == a]
        if len(df) == 0:
            continue
        elif len(df) == 1:
            fest.append(df)
        elif len(df) == 2:
            df = df.drop(df.index[[0]])
            fest.append(df)
        else:
            df = df[df["TestIndex"].isin(["A", "B"])]
            Tl1 = df.iloc[0, 14]
            Tl2 = df.iloc[1, 14]
            if Tl1 > Tl2:
                df = df.drop(df.index[[0]])
                fest.append(df)
            else:
                df = df.drop(df.index[[1]])
                fest.append(df)
    del df
    df = pd.concat(fest, axis=0)
    return df


def get_batchlist():
    path = fr"{os.getcwd()}\batchlist.xlsx"
    try:
        df = pd.read_excel(path, index_col=False)
        df = df.iloc[:, 0]
        return df

    except FileNotFoundError:
        df = pd.DataFrame(columns=["Hier Batches einfügen (A1)"])
        df.to_excel(path, index=False)
        print("Batches in Excel eingeben")


class ShareSetGenerator:
    """Generiert eine Teilgruppenliste (falls nicht vorhanden), die für die Cpk-Analyse benötigt wird.
    counts gibt vor, wie viele Teilgruppen es gibt z.B. 5 Batches in den Daten -> counts = 5
    length gibt vor, wie lang die Liste mit den Teilgruppen werden soll. Dies ist wichtig für die Cpk-Analyse, da
    der Teilgruppensatz nicht länger sein darf, als die Anzahl an Daten.
    alternate   = [1 2 3 1 2 3 1 2 3 ] usw.
    grouped     = [1 1 1 2 2 2 3 3 3 ] usw."""

    def __init__(self, counts, lenght):
        """
        :param counts: Anzahl an Teilgruppen (shared) / Anzahl Datensätze pro Teilgruppe (grouped)
        :type counts: int
        :param lenght: Länge des Datensatzes
        :type lenght: int oder list
        """

        if isinstance(counts, int) is False:
            print("counts must be int")
        else:
            self.count = counts

        if isinstance(lenght, int):
            self.length = lenght
        elif isinstance(lenght, float) or isinstance(lenght, str):
            print("please enter length as int or list")
        else:
            self.length = len(lenght)

    def alternate(self):
        """
        :return: Teilgruppen alternierend [1,2,3,1,2,3 ...]
        :rtype: list
        """
        x = self.count
        i = 1
        length = self.length
        share_set = []
        while len(share_set) < length:
            while i - 1 < x and len(share_set) < length:
                share_set.append(i)
                i += 1
            i = 1

        return share_set

    def grouped(self):
        """
        :return: Teilgruppen gruppiert [1,1,2,2,3,3 ...]
        :rtype: list
        """
        x = self.count
        i = 1
        n = 0
        length = self.length
        share_set = []
        while len(share_set) < length:
            while n < int(length/x) and len(share_set) < length:
                share_set.append(i)
                n += 1
            i += 1
            n = 0

        return share_set

class DataCleaner:

    def __init__(self, data, target_cpk,tol_lower=None, tol_upper=None, norm_dist=True):
        self.data = np.array(data)
        self.mean = np.mean(data)
        self.std = np.std(data)
        self.min = np.min(data)
        self.max = np.max(data)
        self.tol_low = tol_lower
        self.tol_up = tol_upper
        self.target_cpk = target_cpk
        if tol_lower is None and tol_upper is None:
            raise ValueError ("Mindestens eine Toleranz notwendig")

    def cycle_start(self):
        Cpk = igp.cpk_analysis(self.data, tol_up=self.tol_up, tol_low=self.tol_low)[0]
        return Cpk[2]

    def show(self):
        print(self.mean, self.max)


if __name__ == "__main__":
    pass
