# +
# El siguiente codigo permite optener un dataframe con las predicción directamente de un archivo json.
# -

# # Librerias

import re
import json
import pickle
import warnings
import requests
import importlib
import numpy as np
import pandas as pd
import requests, zipfile, io
warnings.filterwarnings('ignore')
from xgboost import plot_importance 
import datetime
import locale
from datetime import date
locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')

# # Ubicación json 

# Accedemos a la base de datos de la que se quiere optener las nuevas predicciones (Si el json se ecuntra descargado, si no se da ninguna ruta se optendra la predicción del dia )
path_1="../Data_Json/Raw-15-11-2022-month/releases_2022_noviembre.json"

# # Cargamos el Modelo

# Cargamos el modelo desde la ubicación del archivo pkl del modelo
ubi_model='xgb_model.pkl'
with open(ubi_model, 'rb') as f:
    xgb_model= pickle.load(f)


def conu(x):
    """
    Recuento de numero de preguntas sin errores
    Return : Int
    """
    try:
        return len(x)
    except:  
        pass


def op_nore(x):
    """
    Numero de preguntas respondidas(se omiten las convalidaciones)
    Return : Int
    """
    re_tex=r'Convalidación |convalidación |CONVALIDA |CONVALIDAR |Convalidacion |convalidacion |CONVALIDACION |NO se presentaron preguntas|convalidación,|NO EXISTE CONVALIDACION|CONVALIDAR|CONVALIDACION|convalidación|convalidar|CONVALIDACIÓN'
    try:
        verfi_colum=pd.DataFrame(x)
        if any(verfi_colum.columns=='description') and  any(verfi_colum.columns=='answer') :
            real_ques=[]
            real_ans=[]
            # display(verfi_colum)
            for i,k in zip(verfi_colum['description'],verfi_colum['answer']):
             if len(re.findall(re_tex,i))>0:
                pass
                # print(i,k)
             else:
                real_ques.append(i) 
                if type(k)==str:
                    real_ans.append(k)
                    # display(df)
                else:
                    # print("no respuesta")
                    real_ans.append(k)

            count_no_asn=real_ans.count(None)
            return count_no_asn

        else:
            return len(verfi_colum['description'])
            pass
    except:
        return np.nan


def cer_39(sd,b):
    """
    Preguntas realizadas fuera del periodo de preguntas
    Return : Int
    """
    out_time=[]
    try:
        en_query=b
        # display(sd)
        stf=pd.DataFrame(sd)
        # display(stf)
        stf['date']=pd.to_datetime(stf['date'])

        for i,k in zip(stf['date'],stf['description']):
            if i<=pd.to_datetime(b) :
                # print(i,"-",b)
                pass
            else:
                if len(re.findall(re_tex,k))==1:
                    pass
                else:
                    out_time.append(len(re.findall(re_tex,k)))
                    # print(i,"-",b,"Mayor",len(re.findall(re_tex,k)))
    except:
        pass
    return len(out_time)


def trasf_data(path="Today",df=False): 
    """
    Genera las variables necesarias para utilizar en el modelo
    return: DataFrame 
    """
    if path!= "Today":
        with open(path,'r', encoding="utf8") as f:
            data = json.loads(f.read())
    elif path=="Today":
        current_month_n=date.today().month
        urljson=f'https://datosabiertos.compraspublicas.gob.ec/PLATAFORMA/download?type=json&year=2022&month={current_month_n}&method=all'
        urlj=urljson
        rj= requests.get(urlj)
        zj = zipfile.ZipFile(io.BytesIO(rj.content))
        name_json=zj.namelist()[0]
        data = json.load(zj.open(name_json))
        # data = json.load(zj.open(path))
    df = pd.json_normalize(data, record_path =['releases'])
    display(df
    df=df[df['tender.procurementMethodDetails']=='Subasta Inversa Electrónica']
    
    for i in df.filter(regex='Date|date', axis=1).columns:
        df[i] = pd.to_datetime(df[i])
    
    df['NF039']=df.apply(lambda x: cer_39(sd=x['tender.enquiries'],b=x['tender.enquiryPeriod.endDate']),axis=1)

    df['enqui_perio']=(df['tender.enquiryPeriod.endDate']-df['tender.enquiryPeriod.startDate']).dt.days

    df['awar_perio']=(df['tender.awardPeriod.endDate']-df['tender.awardPeriod.startDate']).dt.days

    df['Number_ques']=df['tender.enquiries'].apply(conu)

    df['Pre_No_Respu']=df['tender.enquiries'].apply(op_nore)

    df['n_of_Tend']=df['tender.numberOfTenderers']
    df['Year']=df['tender.tenderPeriod.startDate'].dt.year
    df['Codigo']=df['ocid'].str.extract('ocds-5wno2w-(.*\w*.)-|', expand=True)
    return df


def predict_df(path="Today",df=False,model=False,ocid=False,scrap=False):
    """
    Transforma el archivo json, aplica el modelo y seleciona las variables relebante 
    return : Dataframe 
    """
    if path !=False:
        df=trasf_data(path)
    var_tender=['awar_perio','Pre_No_Respu',
           'enqui_perio', 'Number_ques', 'NF039', 'n_of_Tend']
    test_Ndata=df[var_tender]
    # display(df)
    model.get_booster().feature_names=list( test_Ndata.columns)
    # plot_importance(xgb_model, max_num_features=10)
    predict=model.predict(test_Ndata)
    prob=model.predict_proba(test_Ndata)
    test_Ndata["ocid"]=df['ocid']
    test_Ndata["tender.tenderers"]=df["tender.tenderers"]
    test_Ndata["enti_id"]=df["tender.procuringEntity.id"]
    test_Ndata["tender_id"]=df["tender.tenderers"].apply(lambda x: x[0]['id'] if type(x)==list else 0 )
    test_Ndata["predict"]=predict
    test_Ndata["proba"]=prob[:, 1]
    if scrap==True:
        test_Ndata["Estado"]=df["Estado del Proceso"]
        test_Ndata["Link"]=df["Link"]
    test_Ndata.set_index('ocid',inplace=True)
    order_sel=["proba","n_of_Tend",'Pre_No_Respu','awar_perio','enqui_perio', 'Number_ques', 'NF039']
    if ocid!=False:
        test_Ndata=test_Ndata[order_sel]
        selec_ocid=test_Ndata.loc[[ocid]]
        return selec_ocid
    else:
        if scrap==True:
            test_Ndata=test_Ndata[["Estado"]+order_sel+["Link"]]
        else:
            test_Ndata=test_Ndata[order_sel]
        return test_Ndata


df_selec=predict_df(model=xgb_model,ocid=False,scrap=False)

print(df_selec)

# Guardamos la tabla resultante como archivo csv 
df_selec.to_csv('df_predic.csv')  

# +
# Una vez guardada la nueva tabla con las predicciones, si se desea acceder a dichas predicciones mediante el codigo del proceso se puede utilizar el sigueitne codigo
# -

# Buscamos la ubicación del dataframe con las predicciones 
df = pd.read_csv('df_predic.csv')


def search_code(df,code):
    """
    Optiene el codigo del proceso mediante el ocid y busca el codigo requerido
    return : dataframe 
    """
    df['Codigo']=df['ocid'].str.extract('ocds-5wno2w-(.*\w*.)-|', expand=True)
    find=df[df['Codigo']==code]
    json_format=find.to_json(orient = 'records')
    return find


print("Codigo de ejemplo,:SIE-GADMSFP-15-2022 "+"\nSe buscar en la base de datos que ya esta actualida con las predicciones")
code_search="SIE-DPEG-03-2022"
fin_df=search_code(df=df,code=code_search)

print(fin_df)

# +
# Aplicar modelo a un dataframe 
# -

# with open('sie_df.pkl', 'rb') as handle:
    # sie_df = pickle.load(handle)

# +
# df_selec_all_sie=predict_df(df=sel_df_all,model=xgb_model,ocid=False,scrap=False)

# +
# Exportamos los datos con las predición 
# df_selec_all_sie.to_csv('data_sie_predict.csv')  
