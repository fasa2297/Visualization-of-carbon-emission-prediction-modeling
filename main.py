import streamlit as st; 
st.set_page_config(layout="wide")
st.cache_data.clear()
st.markdown(
    """
<style>
    header.css-1avcm0n.e8zbici2{
        backdrop-filter: blur(10px); /* Add blur effect to navbar */
        background-color: rgba(6, 8, 42, 0.5); /* Set transparent background color */
    }
    header.css-18ni7ap.e8zbici2{
        backdrop-filter: blur(10px); /* Add blur effect to navbar */
        background-color: rgba(254, 249, 232, 0.5); /* Set transparent background color */
    }
    div.css-1v0mbdj.etr89bj1 img, div.css-1v0mbdj.ebxwdo61 img{
        background-color: #f5f5f5;
        border: 2px solid;
        border-radius: 10px;
        color: #ffc300;
        box-shadow: 10px;
        pointer-events: none;
        box-shadow: 0 4px 8px 0 rgba(171, 7, 7, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);
    }
    #menu-title{
        color: #ffc300;
    }
    div.mapboxgl-map{
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(255, 241, 221, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);
    }
</style>
""",
    unsafe_allow_html=True,
)

import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html

df_data = pd.read_csv('data/DataSumatera_land_98u22.csv')
def set_data():
    st.subheader("📁Set Data")
    expn_data = st.expander("🔔 Tentang Data")
    expn_data.caption("Dalam proses prediksi nilai emisi karbon penyebab hotspot kebakaran hutan menggunakan dataset emisi karbon dan indikator iklim, yang terdiri dari data Emisi Karbon, Curah Hujan, Kelembapan, Kecepatan Angin, Rata-rata Suhu, dan Suhu Maksimum, yang bersumber dari gabungan data (GFED4.1s) Global Fire and Emissions Data dan (ERA5). Tugas Akhir ini menggunakan data bulanan dengan record data pada rentang tahun1998 hingga 2022. Untuk data Emisi Karbon bersumber dari (GFED4.1s) yang menggunakan grid data yang beresolusi 0,25° x 0,25°, dengan setiap file data berisi 1440 kolom dan 720 baris, dengan unit satuan emisi bulanan gC/m^2. Dataset (GFED4.1s) dapat diperoleh dari https://www.geo.vu.nl/~gwerf/GFED/GFED4/. Kemudian untuk data iklim seperti tp (Curah Hujan), d2m (Kelembapan), si10 (Kecepatan Angin), t2m (Rata-rata Suhu), dan tmax (Suhu Maksimum), menggunakan data dari (ERA5) atau generasi kelima dari data analisis ulang Prakiraan Cuaca Jangka Menengah Eropa atau European Centre Medium-Range Weather Forecasts (ECMWF). Data pada ERA5 telah di-regrid ke grid lat-lon reguler beresolusi 0,25° x 0,25°. Dataset iklim ERA5 digunakan didapatkan dari https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means. Data Emisi Kabon GFED4.1 menjadi target prediksi data untuk training dan testing model Random Forest Regression dan Gradient Boosting Regression.\n\n⚠️Keterangan Variabel: \n Terdapat 9 atribut utama dari dataset diantaranya adalah:\n 1. time (dates)\n 2. latitude (latitude)\n 3. longitude (longitude)\n 4. si10 (wind speed)\n 5. d2m (humidity)\n 6. t2m (average temperature)\n 7. tp (precipitation)\n 8. tmax (temperature maximum)\n 9. emissions (emissions carbon)")    
    
    st.subheader("Sampel Set Data :")
    st.caption("Menampilkan sampel set data dari 8 index data teratas dan 8 index data terakhir")
    st.dataframe(df_data.head(8))
    st.dataframe(df_data.tail(8))
    st.caption(df_data.shape)

    st.subheader("Statistik Data :")
    st.caption("Menampilkan statistik deskriptif dari set data")
    df_describe = df_data.describe().fillna("").astype("str")
    st.write(df_describe)

def visualisasi_data():
    st.subheader("🎨Visualisasi Set Data")

    # Display the map in Streamlit
    st.subheader("🗺️ Sampel Bulanan Sebaran Emisi Karbon GFED4.1s :")

    time = '2022-12-01'
    col1, col2, col3 = st.columns([1,1,1])
    with col1: st.error('Pilih Tanggal Tahun dan Bulan, Rentang Tahun 1998-2022')
    with col2: option_ye = st.selectbox('Tahun:',
                            ('1998', '1999', '2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021','2022'))
    with col3: option_mo = st.selectbox('Bulan:',
                            ('01','02', '03','04','05','06','07','08','09','10','11','12'))

    col1_cpt, col2_cpt = st.columns([0.7,3])
    with col1_cpt: 
        if st.button('🔍Tampilkan'): time = option_ye+"-"+option_mo+"-"+'01'
    with col2_cpt: st.write('( Saat ini memvisualisasikan data tahun dan bulan',time,')') 
    
    Sampel_emissions_mo22 = df_data.loc[(df_data['time'] >= time)& (df_data['time'] <= time)][['latitude', 'longitude', 'emissions']]
    Sampel_emissions_mo22.replace(0, np.nan, inplace=True)
    Sampel_emissions_mo22.dropna(inplace=True)
    st.map(Sampel_emissions_mo22)
    
    st.subheader("📊Distribusi Nilai")
    st.caption("Menampilkan distribusi nilai per-tingkat figur dari features utama")
    st.image('data/distplot.png')

    st.subheader("📊Boxplot Data")
    st.caption("Menampilkan distribusi nilai minimum, persentil ke-25 (25% data), median (50% data), persentil ke-75 (75% data), nilai maksimum serta kemungkinan keberadaan outlier dari features utama")
    st.image('data/boxplot.png')

    st.subheader("📊Total Emisi Karbon 1998-2022")
    st.caption("Menampilkan tingkat total emisi karbon versi GFED4.1s per-tahun selama 1998-2022")
    st.image('data/Total_GFED.png')
    st.caption("Menampilkan tingkat total emisi karbon 3 terbesar versi GFED4.1s")
    st.image('data/3big_GFED.png')

    st.subheader("📄Korelasi Variabel")
    corrtab1, corrtab2 = st.tabs(["Cross-Correlation Analysis", "Pearson Correlation"])
    with corrtab1: 
        st.caption("Menampilkan korelasi variabel terhadap emisi karbon menggunakan metode Cross-Correlation Analysis")
        st.image('data/Corellation_cross.png', caption="Phone&Tablet: tab 2x -> swap ke kiri -> view fullscreen memperbesar gambar")
    with corrtab2: 
        st.caption("Menampilkan korelasi variabel terhadap emisi karbon menggunakan metode Pearson Correlation")
        st.image('data/Corellation_pearson.png')

def emissions_display_rfr_default():
        st.subheader("🔥Visualisasi Prediksi Random Forest Regression")
        col1, col2 = st.columns([2.5,1])
        with col1:
            st.info("Menampilkan proyeksi prediksi sebaran emisi karbon pulau Sumatera untuk tahun 2021, 2022, 2023, sebagai indikasi hotspot kebarakaran hutan hasil Randon Forest Regression dengan rasio **88% : 12% (3 Tahun)**. Dan menampilkan Kontribusi Variabel dan Skor WOE terhadap model RFR.", icon="ℹ️")
        with col2:
            st.warning("RMSE: 10,43 MAE: 108,91", icon="⚠️")
 
        rtab1, rtab2, rtab3 = st.tabs(["Emisi Karbon 2021", "Emisi Karbon 2022", "Emisi Karbon 2023"])
        with rtab1: 
            option = st.selectbox( '🔻Predisi/Aktual', ('Prediksi Emisi Karbon 2021', 'Aktual Emisi Karbon 2021'))
            if option == "Prediksi Emisi Karbon 2021": st.image('data/rfr_em21.png')
            elif option == "Aktual Emisi Karbon 2021": st.image('data/aktual/Aktual_em_2021.png')
        with rtab2: 
            option = st.selectbox( '🔻Predisi/Aktual', ('Prediksi Emisi Karbon 2022', 'Aktual Emisi Karbon 2022'))
            if option == "Prediksi Emisi Karbon 2022": st.image('data/rfr_em22.png')
            elif option == "Aktual Emisi Karbon 2022": st.image('data/aktual/Aktual_em_2022.png')
        with rtab3: 
            option = st.selectbox( '🔻Predisi/Aktual', ('Prediksi Emisi Karbon 2023', 'Aktual Emisi Karbon 2023'))
            if option == "Prediksi Emisi Karbon 2023": st.image('data/rfr_em23.png')
            elif option == "Aktual Emisi Karbon 2023": st.image('data/aktual/Aktual_em_2023.png')

        st.subheader("🔥Kontribusi Variabel Terhadap Random Forest Regression")
        st.image('data/rfr_varcontribution.png')

        st.subheader("🔥Skor WOE Random Forest Regression")
        st.image('data/woe_rfr.png')

def emissions_display_gbr_default():
        st.subheader("🔥Visualisasi Prediksi Gradient Boosting Regression")
        col1, col2 = st.columns([2.5,1])
        with col1:
            st.info("Menampilkan prediksi sebaran emisi karbon pulau Sumatera untuk tahun 2021, 2022, 2023, sebagai indikasi hotspot kebarakaran hutan hasil Gradient Boosting Regression dengan rasio **88% : 12% (3 Tahun)**. Dan menampilkan Kontribusi Variabel dan Skor WOE terhadap model GBR.", icon="ℹ️")
        with col2:
            st.warning("RMSE: 10,87 MAE: 2,91", icon="⚠️")
        
        gtab1, gtab2, gtab3 = st.tabs(["Emisi Karbon 2021", "Emisi Karbon 2022", "Emisi Karbon 2023"])
        with gtab1: 
            option = st.selectbox( '🔻Predisi/Aktual', ('Prediksi Emisi Karbon 2021', 'Aktual Emisi Karbon 2021'))
            if option == "Prediksi Emisi Karbon 2021": st.image('data/gbr_em21.png')
            elif option == "Aktual Emisi Karbon 2021": st.image('data/aktual/Aktual_em_2021.png')
        with gtab2: 
            option = st.selectbox( '🔻Predisi/Aktual', ('Prediksi Emisi Karbon 2022', 'Aktual Emisi Karbon 2022'))
            if option == "Prediksi Emisi Karbon 2022": st.image('data/gbr_em22.png')
            elif option == "Aktual Emisi Karbon 2022": st.image('data/aktual/Aktual_em_2022.png')
        with gtab3: 
            option = st.selectbox( '🔻Predisi/Aktual', ('Prediksi Emisi Karbon 2023', 'Aktual Emisi Karbon 2023'))
            if option == "Prediksi Emisi Karbon 2023": st.image('data/gbr_em23.png')
            elif option == "Aktual Emisi Karbon 2023": st.image('data/aktual/Aktual_em_2023.png')

        st.subheader("🔥Kontribusi Variabel Terhadap Gradient Boosting Regression")
        st.image('data/gbr_varcontribution.png')

        st.subheader("🔥Skor WOE Random Forest Regression")
        st.image('data/woe_gbr.png')

def emissions_display_rfr_r4():
        st.subheader("🔥Visualisasi Prediksi Random Forest Regression")
        col1, col2 = st.columns([2.5,1])
        with col1:
            st.info("Menampilkan proyeksi prediksi sebaran emisi karbon pulau Sumatera untuk tahun 2020, 2021, 2022, 2023, sebagai indikasi hotspot kebarakaran hutan hasil Randon Forest Regression dengan rasio **84% : 16% (4 Tahun)**.", icon="ℹ️")
        with col2:
            st.warning("RMSE: 128.80 MAE: 16590.48", icon="⚠️")
        rtab1, rtab2, rtab3, rtab4 = st.tabs(["Emisi Karbon 2020", "Emisi Karbon 2021", "Emisi Karbon 2022", "Emisi Karbon 2023"])
        with rtab1: 
            option = st.selectbox( '🔻Predisi/Aktual', ('Prediksi Emisi Karbon 2020', 'Aktual Emisi Karbon 2020'))
            if option == "Prediksi Emisi Karbon 2020": st.image('data/rfr-ratio/rfr_em20_ratio_4.png')
            elif option == "Aktual Emisi Karbon 2020": st.image('data/aktual/Aktual_em_2020.png')
        with rtab2:
            option = st.selectbox( '🔻Predisi/Aktual', ('Prediksi Emisi Karbon 2021', 'Aktual Emisi Karbon 2021'))
            if option == "Prediksi Emisi Karbon 2021": st.image('data/rfr-ratio/rfr_em21_ratio_4.png')
            elif option == "Aktual Emisi Karbon 2021": st.image('data/aktual/Aktual_em_2021.png')
        with rtab3:
            option = st.selectbox( '🔻Predisi/Aktual', ('Prediksi Emisi Karbon 2022', 'Aktual Emisi Karbon 2022'))
            if option == "Prediksi Emisi Karbon 2022": st.image('data/rfr-ratio/rfr_em22_ratio_4.png')
            elif option == "Aktual Emisi Karbon 2022": st.image('data/aktual/Aktual_em_2022.png')
        with rtab4:
            option = st.selectbox( '🔻Predisi/Aktual', ('Prediksi Emisi Karbon 2023', 'Aktual Emisi Karbon 2023'))
            if option == "Prediksi Emisi Karbon 2023": st.image('data/rfr-ratio/rfr_em23_ratio_4.png')
            elif option == "Aktual Emisi Karbon 2023": st.image('data/aktual/Aktual_em_2023.png')

def emissions_display_gbr_r4():
        st.subheader("🔥Visualisasi Prediksi Gradient Boosting Regression")
        col1, col2 = st.columns([2.5,1])
        with col1:
            st.info("Menampilkan prediksi sebaran emisi karbon pulau Sumatera untuk tahun 2020, 2021, 2022, 2023, sebagai indikasi hotspot kebarakaran hutan hasil Gradient Boosting Regression dengan rasio **84% : 16% (4 Tahun)**.", icon="ℹ️")
        with col2:
            st.warning("RMSE: 129.66 MAE: 9.86", icon="⚠️")

        rtab1, rtab2, rtab3, rtab4 = st.tabs(["Emisi Karbon 2020", "Emisi Karbon 2021", "Emisi Karbon 2022", "Emisi Karbon 2023"])
        with rtab1:
            option = st.selectbox( '🔻Predisi/Aktual', ('Prediksi Emisi Karbon 2020', 'Aktual Emisi Karbon 2020'))
            if option == "Prediksi Emisi Karbon 2020": st.image('data/gbr-ratio/gbr_em20_ratio_4.png')
            elif option == "Aktual Emisi Karbon 2020": st.image('data/aktual/Aktual_em_2020.png')
        with rtab2:
            option = st.selectbox( '🔻Predisi/Aktual', ('Prediksi Emisi Karbon 2021', 'Aktual Emisi Karbon 2021'))
            if option == "Prediksi Emisi Karbon 2021": st.image('data/gbr-ratio/gbr_em21_ratio_4.png')
            elif option == "Aktual Emisi Karbon 2021": st.image('data/aktual/Aktual_em_2021.png')
        with rtab3:
            option = st.selectbox( '🔻Predisi/Aktual', ('Prediksi Emisi Karbon 2022', 'Aktual Emisi Karbon 2022'))
            if option == "Prediksi Emisi Karbon 2022": st.image('data/gbr-ratio/gbr_em22_ratio_4.png')
            elif option == "Aktual Emisi Karbon 2022": st.image('data/aktual/Aktual_em_2022.png')        
        with rtab4:
            option = st.selectbox( '🔻Predisi/Aktual', ('Prediksi Emisi Karbon 2023', 'Aktual Emisi Karbon 2023'))
            if option == "Prediksi Emisi Karbon 2023": st.image('data/gbr-ratio/gbr_em23_ratio_4.png')
            elif option == "Aktual Emisi Karbon 2023": st.image('data/aktual/Aktual_em_2023.png')

def emissions_display_rfr_r5():
        st.subheader("🔥Visualisasi Prediksi Random Forest Regression")
        col1, col2 = st.columns([2.5,1])
        with col1:
            st.info("Menampilkan proyeksi prediksi sebaran emisi karbon pulau Sumatera untuk tahun 2019, 2020, 2021, 2022, 2023, sebagai indikasi hotspot kebarakaran hutan hasil Randon Forest Regression dengan rasio **80% : 20% (5 Tahun)**.", icon="ℹ️")
        with col2:
            st.warning("RMSE: 115.89 MAE: 13430.80", icon="⚠️")

        rtab1, rtab2, rtab3, rtab4, rtab5 = st.tabs(["Emisi Karbon 2019", "Emisi Karbon 2020", "Emisi Karbon 2021", "Emisi Karbon 2022", "Emisi Karbon 2023"])
        with rtab1:
            option = st.selectbox( '🔻Predisi/Aktual', ('Prediksi Emisi Karbon 2019', 'Aktual Emisi Karbon 2019'))
            if option == "Prediksi Emisi Karbon 2019": st.image('data/rfr-ratio/rfr_em19_ratio_5.png')
            elif option == "Aktual Emisi Karbon 2019": st.image('data/aktual/Aktual_em_2019.png')
        with rtab2: 
            option = st.selectbox( '🔻Predisi/Aktual', ('Prediksi Emisi Karbon 2020', 'Aktual Emisi Karbon 2020'))
            if option == "Prediksi Emisi Karbon 2020": st.image('data/rfr-ratio/rfr_em20_ratio_5.png')
            elif option == "Aktual Emisi Karbon 2020": st.image('data/aktual/Aktual_em_2020.png')
        with rtab3:
            option = st.selectbox( '🔻Predisi/Aktual', ('Prediksi Emisi Karbon 2021', 'Aktual Emisi Karbon 2021'))
            if option == "Prediksi Emisi Karbon 2021": st.image('data/rfr-ratio/rfr_em21_ratio_5.png')
            elif option == "Aktual Emisi Karbon 2021": st.image('data/aktual/Aktual_em_2021.png')
        with rtab4:
            option = st.selectbox( '🔻Predisi/Aktual', ('Prediksi Emisi Karbon 2022', 'Aktual Emisi Karbon 2022'))
            if option == "Prediksi Emisi Karbon 2022": st.image('data/rfr-ratio/rfr_em22_ratio_5.png')
            elif option == "Aktual Emisi Karbon 2022": st.image('data/aktual/Aktual_em_2022.png')
        with rtab5:
            option = st.selectbox( '🔻Predisi/Aktual', ('Prediksi Emisi Karbon 2023', 'Aktual Emisi Karbon 2023'))
            if option == "Prediksi Emisi Karbon 2023": st.image('data/rfr-ratio/rfr_em23_ratio_5.png')
            elif option == "Aktual Emisi Karbon 2023": st.image('data/aktual/Aktual_em_2023.png')


def emissions_display_gbr_r5():
        st.subheader("🔥Visualisasi Prediksi Gradient Boosting Regression")
        col1, col2 = st.columns([2.5,1])
        with col1:
            st.info("Menampilkan prediksi sebaran emisi karbon pulau Sumatera untuk tahun 2019, 2020, 2021, 2022, 2023, sebagai indikasi hotspot kebarakaran hutan hasil Gradient Boosting Regression dengan rasio **80% : 20% (5 Tahun)**.", icon="ℹ️")
        with col2:
            st.warning("RMSE: 116.56 MAE: 9.60", icon="⚠️")

        rtab1, rtab2, rtab3, rtab4, rtab5 = st.tabs(["Emisi Karbon 2019", "Emisi Karbon 2020", "Emisi Karbon 2021", "Emisi Karbon 2022", "Emisi Karbon 2023"])
        with rtab1:
            option = st.selectbox( '🔻Predisi/Aktual', ('Prediksi Emisi Karbon 2019', 'Aktual Emisi Karbon 2019'))
            if option == "Prediksi Emisi Karbon 2019": st.image('data/gbr-ratio/gbr_em19_ratio_5.png')
            elif option == "Aktual Emisi Karbon 2019": st.image('data/aktual/Aktual_em_2019.png')
        with rtab2: 
            option = st.selectbox( '🔻Predisi/Aktual', ('Prediksi Emisi Karbon 2020', 'Aktual Emisi Karbon 2020'))
            if option == "Prediksi Emisi Karbon 2020": st.image('data/gbr-ratio/gbr_em20_ratio_5.png')
            elif option == "Aktual Emisi Karbon 2020": st.image('data/aktual/Aktual_em_2020.png')
        with rtab3:
            option = st.selectbox( '🔻Predisi/Aktual', ('Prediksi Emisi Karbon 2021', 'Aktual Emisi Karbon 2021'))
            if option == "Prediksi Emisi Karbon 2021": st.image('data/gbr-ratio/gbr_em21_ratio_5.png')
            elif option == "Aktual Emisi Karbon 2021": st.image('data/aktual/Aktual_em_2021.png')
        with rtab4:
            option = st.selectbox( '🔻Predisi/Aktual', ('Prediksi Emisi Karbon 2022', 'Aktual Emisi Karbon 2022'))
            if option == "Prediksi Emisi Karbon 2022": st.image('data/gbr-ratio/gbr_em22_ratio_5.png')
            elif option == "Aktual Emisi Karbon 2022": st.image('data/aktual/Aktual_em_2022.png')
        with rtab5:
            option = st.selectbox( '🔻Predisi/Aktual', ('Prediksi Emisi Karbon 2023', 'Aktual Emisi Karbon 2023'))
            if option == "Prediksi Emisi Karbon 2023": st.image('data/gbr-ratio/gbr_em23_ratio_5.png')
            elif option == "Aktual Emisi Karbon 2023": st.image('data/aktual/Aktual_em_2023.png')

def var_contributions():
    st.header('Kontribusi Variabel untuk Model')
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔥Random Forest Regression")
        st.image('data/rfr_varcontribution.png')
            
    with col2:
        st.subheader("🔥Gradient Boosting Regression")
        st.image('data/gbr_varcontribution.png')
    
def emissions_total():
    st.header('Total Prediksi Emisi Karbon')
    compare_emission_21_rfr_gbr = pd.DataFrame({ "Months 2021": ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
                                                "Actual" :     ['423.11','388.67','1388.40','476.10','145.48','113.03','171.02','262.66','183.69','142.76','70.80','35.58'],
                                                "Predict RFR": ['2158.55','2463.04','4001.55','833.57','1828.85','1494.72','1184.47','8423.53','1966.60','1482.33','310.72','847.27'],
                                                "Predict GBR": ['2188.50','2071.59','4446.14','2136.40','4268.14','1663.70','1539.36','6922.77','1813.56','1560.82','1024.97','1229.25']})
    compare_emission_22_rfr_gbr = pd.DataFrame({ "Months 2022": ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
                                                "Actual" :     ['28.83','234.61','406.35','201.89','184.84','220.16','522.79','389.67','331.56','306.79','87.20','85.85'],
                                                "Predict RFR": ['769.48','4397.96','591.83','556.57','1096.36','1988.88','3738.69','1368.42','1201.88','3891.67','1118.04','410.82'],
                                                "Predict GBR": ['1115.90','4727.73','1130.21','1405.63','2403.65','2319.90','3072.10','1236.27','1250.39','3535.08','1505.12','1025.74']})
    compare_emission_23_rfr_gbr = pd.DataFrame({ "Months 2023": ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
                                                "Actual" :     ['103.82','126.31','245.93','356.24','228.33','107.07','249.51','258.07','206.81','21.36','60.07','20.93'],
                                                "Predict RFR": ['828.67','925.03','2086.96','729.73','3440.22','578.19','3497.18','912.57','1158.61','435.70','432.81','419.79'],
                                                "Predict GBR": ['1405.76','1398.19','2538.47','1758.95','5918.13','1030.49','3338.24','1054.02','1209.52','1034.49','1073.38','990.79']})

    ttab1, ttab2, ttab3 = st.tabs(["📈 Total Emisi Karbon 2021", "📈 Total Emisi Karbon 2022", "📈 Total Emisi Karbon 2023"])
    with ttab1: 
        st.image('data/total_em21.png')
        st.dataframe(compare_emission_21_rfr_gbr)
    with ttab2: 
        st.image('data/total_em22.png')
        st.dataframe(compare_emission_22_rfr_gbr)
    with ttab3: 
        st.image('data/total_em23.png')
        st.dataframe(compare_emission_23_rfr_gbr)
    
def emissions_error():
    st.header('Total Error Prediksi Emisi Karbon')
    compare_emission_error_rfr = pd.DataFrame({ "Months": ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
                                                "Error 2021": ['1735.44','2074.36','2613.15','357.47','1683.37','1381.69','1013.44','8160.87','1782.91','1339.58','239.91','811.68'],
                                                "Error 2022": ['740.65','4163.35','185.48','354.68','911.53','1768.72','3215.90','978.74','870.32','3584.88','1030.84','324.97'],
                                                "Error 2023": ['724.84','798.72','1841.03','373.48','3211.90','471.12','3247.67','654.51','951.79','414.34','372.73','398.86']
    })
    compare_emission_error_gbr = pd.DataFrame({ "Months": ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
                                                "Error 2021": ['1765.39','1682.91','3057.74','1660.29','4122.67','1550.68','1368.34','6660.11','1629.86','1418.06','954.17','1193.66'],
                                                "Error 2022": ['1087.06','4493.12','723.86','1203.74','2218.81','2099.74','2549.31','846.59','918.83','3228.29','1417.93','939.89'],
                                                "Error 2023": ['1301.94','1271.88','2292.55','1402.71','5689.81','923.42','3088.73','795.96','1002.71','1013.13','1013.30','969.87']
    })

    etab1, etab2 = st.tabs(["🔥Total Error Random Forest Regression", "🔥Total Error Gradient Boosting Regression"])
    with etab1: 
        st.image('data/rfr_error.png')
        st.dataframe(compare_emission_error_rfr)
    with etab2: 
        st.image('data/gbr_error.png')
        st.dataframe(compare_emission_error_gbr)

def ratio_split_data():
    ratio = st.radio(
        "🔻Rasio Data Training-Testing Terhadap Model Prediksi:",
        ('88% : 12% (2021-2023) Default' , '86% : 16% (2020-2023)', '80% : 20% (2019-2023)'))
    if ratio == '86% : 16% (2020-2023)':
        option = st.selectbox( '🔻Model Machine Learning', ('Random Forest Regression', 'Gradient Boosting Regeression'))
        st.markdown("""---""")
        if   option == "Random Forest Regression":emissions_display_rfr_r4()
        elif option == "Gradient Boosting Regeression": emissions_display_gbr_r4()
    elif ratio == '80% : 20% (2019-2023)':
        option = st.selectbox( '🔻Model Machine Learning', ('Random Forest Regression', 'Gradient Boosting Regeression'))
        st.markdown("""---""")
        if   option == "Random Forest Regression":emissions_display_rfr_r5()
        elif option == "Gradient Boosting Regeression": emissions_display_gbr_r5()
    else:
        option = st.selectbox( '🔻Model Machine Learning', ('Random Forest Regression', 'Gradient Boosting Regeression'))
        st.markdown("""---""")
        if   option == "Random Forest Regression":emissions_display_rfr_default()
        elif option == "Gradient Boosting Regeression": emissions_display_gbr_default()

#--------------------------------------------------------------------------------------------------------------------------------------
def menu():
    with st.sidebar:
        choose = option_menu("Prediksi Emisi Karbon Sumatera", ["Set Data",
                                                                      "Visualisasi Data",
                                                                      "Evaluasi Prediksi", 
                                                                      "Total Prediksi Emisi Karbon", 
                                                                      "Total Eror Prediksi Emisi Karbon",],
                            menu_icon="app-indicator", default_index=0,
                            styles={
            "menu-title":           {"color": "#EC7063"},
            "container":            {"background-color": "#fafafa"},
            "icon":                 {"color": "orange", "font-size": "15px"}, 
            "nav-link":             {"color": "#F4511E","font-size": "14px", "text-align": "left", "margin":"0px", "--hover-color": "#eee","border-radius": "10px"},
            "nav-link active":      {"text-shadow": "2px 2px 4px #000000"},
            "nav-link-selected":    {"background-color": "#FBEEE6"},
        }
        )
        expn_a = st.expander("📝 Author")
        expn_a.caption("2023\n\nFauzi Arya Surya Abadi (S1 Informatika)")
        expn_b = st.expander("🔔 About")
        expn_b.caption("Tujuan penelitian ini secara umum untuk mengetahui hasil output prediksi data hotspot 2021, 2022, 2023 berupa nilai dan visualisasi hasil dari penggunaan Machine Learning dengan model Random Forest Regression dan dan Gradient Boosting Regression, dengan kasus prediksi hotspot yang dihasilkan untuk wilayah Sumatra. Sehingga penulis dapat memberikan tujuan penelitian sebagai berikut:\n 1.	Membandingkan algoritma Supervised Learning model Random Forest Regression dan Gradient Boosting Regression dalam memprediksi hotspots kebakaran hutan untuk studi kasus wilayah Sumatra. \n2.	Mengetahui indikator iklim apa saja yang memilki kolerasi paling tinggi terhadap penyebaran hotspot, dan persentase kontribusi dalam model. \n3.	Menggunakan bahasa pemorgraman python untuk implementasi memprediksi hotspot menggunakan model Random Forest Regression dan Gradient Boosting Regression.")

    if choose   == "Set Data": set_data()
    elif choose == "Visualisasi Data": visualisasi_data()
    elif choose == "Evaluasi Prediksi": ratio_split_data()
    elif choose == "Total Prediksi Emisi Karbon": emissions_total()
    elif choose == "Total Eror Prediksi Emisi Karbon": emissions_error()

if __name__ == "__main__":
    menu()