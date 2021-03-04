

import pandas as pd
import numpy  as np

import osr,ogr

### Class ###



class Layer_Engine():

    def __init__(self,data,columns):

        self.data            = data
        self.len_columns     = len(columns)
        self.df              = pd.DataFrame(data = self.data, columns = columns)
        self.len_rows        = self.df.shape[0]
        self.columns         = columns

        self.df_count = False
        

    def Create_CSV(self,csv_name,set_index = ''):
        if set_index:
            df2 = self.df.set_index(set_index)
            return df2.to_csv(csv_name)
        self.df.to_csv(csv_name)


    def drop_fields(self,fields_list):
        self.df.drop(columns = fields_list, axis=1)

    def drop_nulls(self,field):
        self.df = self.df.dropna(how = 'any',subset=[field])

    def Sum(self,field):
        return self.df[field].sum()


    def Get_min_max_ofGroup(self,GroupingField,SearcField):
        gb_obj            = self.df.groupby (by = GroupingField)
        df_min            = gb_obj.agg     ({SearcField : np.min})
        df_max            = gb_obj.agg     ({SearcField : np.max})
        df_edge           = pd.concat      ([df_min,df_max])
        df2               = pd.merge       (self.df,df_edge, how='inner', on='index1')
        return df2


    def Count_field(self,field):
        self.df['count'] = self.df.groupby(field)[field].transform('count')


    def Len_field(self,field,as_int = False):

        if as_int:
            len_field = self.df[field].apply(str).apply(len).astype(int)
            if len_field.shape[0] > 1:
                len_field = len_field[0]
            return int(len_field)
        else:
            self.df[field + '_len'] = self.df[field].apply(len)

    def Filter_df(self,field,Value,Update_df = False):
        if Update_df:
            self.df = self.df[self.df[field] == Value]
        else:
            df_filter = self.df[self.df[field] == Value]
            return df_filter


    def Groupby_and_count(self,field,name_field_count = ''):

        if name_field_count == '':
            name_field_count = str(field) + "_num"
        count_data    = self.df.groupby(field).size()
        count_data    = count_data.to_frame().reset_index()
        self.df_count = count_data


    def Dict(self,index_key):

        dict_  = self.df.set_index(index_key)
        dict_2 = dict_.T.to_dict()
        return dict_2

    def del_null(self,field):
        self.df = self.df[self.df[field].isnull()]

    def del_if_in(self,field,df_or_list,reverse = False):
        ''' df_or_list = result['index1']'''
        if reverse:
            self.df = self.df.loc[~self.df[field].isin(df_or_list)]

    def Group_and_Rank(self,GroupField,RankField,first_rank = True,Update_df = False):
        df2 = self.df.copy()
        df2["RANK"] = self.df.groupby(GroupField)[RankField].rank(method='first',ascending=False)
        if first_rank:
            df2     = self.df[self.df['RANK'] == 1]
        if Update_df:
            self.df = df2

    def coord_from_WGS84_to_IsraelUTM(self,field_X,field_Y):
        self.df['X_Y']  = self.df.apply(lambda row: get_proj_osr(row[field_X] , row[field_Y]), axis=1)

    def return1_if_in_list(self,new_field,check_field,listValues):
        self.df[new_field] = self.df.apply(lambda row: check_syn(row[check_field],listValues), axis=1)



#####  Func   ####


def check_syn(value,list_check):
    '''
    list_check = ['חבד','חב"ד','בי"כ','בני ברק','חסידי','בית מדרש',"בית כנסת",'תורה','ישיבה','בית הכנסת','ישיבת']
    '''
    a = [i for i in list_check if value if i in value]
    if a:
        return 1


def get_proj_osr(pointX,pointY):
    inputEPSG = 4326
    outputEPSG = 2039
    
    # create a geometry from coordinates
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(pointX, pointY)
    
    # create coordinate transformation
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(inputEPSG)
    
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(outputEPSG)
    
    coordTransform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
    
    # transform point
    point.Transform(coordTransform)
    
    # print point in EPSG 4326
    return str(point.GetX()) +'-'+ str(point.GetY())


def read_excel_sheets(path2):
    x1 = pd.ExcelFile(path2)
    df = pd.DataFrame()
    columns = None
    for idx,name in enumerate(x1.sheet_names):
        try:
            sheet = x1.parse(name)
            if idx == 0:
                columns = sheet.columns
            sheet.columns = columns
        except:
            print ("coudent read sheet {}".format(name))
        df = df.append(sheet,ignore_index = True)
            
    return df
