import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from PIL import Image
import seaborn as sns
import io

st.markdown(
    """
    <style>
        .stApp {
            background-color: white !important;
        }
        body, .stMarkdown, .stTextInput, .stButton, .stSelectbox, .stSlider, .stDataFrame, .stTable {
            color: black !important;
        }
        h1, h2, h3, h4, h5, h6, .stHeader, .stSubheader {
            color: black !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("Teknik Iterasi pada Matriks Aljabar: Fast Fourier Transform")

st.markdown("""
- Nama : Naufal Fadhlullah
- NIM : 20234920001""")

st.header("Flowchart")
st.image("https://raw.githubusercontent.com/naaufald/METNUM/main/flowchart.png", caption="Diagram Flowchart")

st.header("Data : dataset yang digunakan berisi data iklim yang mencakup beberapa variabel. dataset ini memiliki 12 kolom dan 589,265 baris.")
st.image("https://raw.githubusercontent.com/naaufald/METNUM/main/datanya.png", caption = "Sumber Data: https://www.kaggle.com/datasets/greegtitan/indonesia-climate?select=climate_data.csv")
st.markdown("""
dataset yang digunakan berisi data iklim yang mencakup beberapa variabel. dataset ini memiliki 12 kolom dan 589,265 baris.
            berikut variabel yang ada di data:
1. **Tn** – Temperatur Minimum
2. **Tx** – Temperatur Maksimum.
3. **Tavg** – Temperatur rata-rata.
4. **RH_avg** – average humidity.
5. **RR** – Rainfall (mm)
6. **ss** – durasi matahari (jam)
7. **ff_x** – Max wind speed.
8. **ddd_x** – wind direction at maximum speed.
9. **ff_avg** – average wind speed.
10. **ddd_car** – most wind direction.
11. **station_id** – station id which record the data.
namun, karena 589.265 baris data terlalu banyak, saya memotong hanya sampai 1096 baris data.""")

st.subheader("Eksplorasi Data")
url = "https://raw.githubusercontent.com/naaufald/METNUM/main/climate_data.csv"
st.write("Data iklim Indonesia")
df = pd.read_csv(url)
st.dataframe(df)

st.write("Distribusi data, missing values, dan outliers")
st.write("### Tampilkan & Analisis Data")
st.write(df.head())
st.write("**Statistik Deskriptif:**")
st.write(df.describe())
st.write("**Missing Values:**")
st.write(df.isnull().sum())

st.subheader("Visualisasi")
kode_r = """
```r
data$date <- as.Date(data$date, format="%d-%m-%Y")
ggplot(data, aes(x = date, y = Tavg)) +
  geom_line(color = "blue") +
  labs(title = "Tren Suhu Rata-rata", x = "Tanggal", y = "Suhu Rata-rata (°C)") +
  theme_minimal()"""
st.markdown(kode_r)
st.image("https://raw.githubusercontent.com/naaufald/METNUM/main/tren_suhu.png")

kode_r = """
```r
ggplot(data, aes(x = RR)) +
  geom_histogram(binwidth = 10, fill = "red", color = "black") +
  labs(title = "Distribusi Curah Hujan", x = "Curah Hujan (mm)", y = "Frekuensi") +
  theme_minimal()"""
st.markdown(kode_r)
st.image("https://raw.githubusercontent.com/naaufald/METNUM/main/curah_hujan.png")

st.subheader("Feature Engineering")
st.write("Identifikasi Fitur")
kode_r = """
```r
cor_matrix <- cor(data[, sapply(data, is.numeric)], use = "complete.obs")
cor_df <- as.data.frame(as.table(cor_matrix))
cor_df <- cor_df[abs(cor_df$Freq) >= 0.50 & cor_df$Var1 != cor_df$Var2, ]
print(cor_df)"""
st.markdown(kode_r)
st.image("https://raw.githubusercontent.com/naaufald/METNUM/main/korelasi.png")
st.markdown("""
pada data yang digunakan dan menggunakan kode R untuk melihat korelasi antar variabel dan berpacu pada Pearson, data yang kuat memiliki korelasi rentang 0.50--0.70 (korelasi kuat), didapatkan bahwa variabel yang memiliki korelasi tinggi ada pada variabel
Tavg x Tn, dan juga Tavg x Tx, maka Tx atau Tn akan dihapus karena informasinya mirip dan akan menghapus RR karena memiliki banyak NA (125384)""")

kode_r = """
```r
data <- data %>% select(-Tn, -Tx, -RR, -station_id)"""
st.markdown(kode_r)
st.image("https://raw.githubusercontent.com/naaufald/METNUM/main/delete.png")

st.subheader("Transformasi data")
st.markdown("""karena data banyak yang NA, maka akan di ganti nilainya menggunakan median""")
kode_r = """
```r
data <- data.frame(lapply(data, function(x) ifelse(is.na(x), median(x, na.rm = TRUE), x)))"""
st.markdown(kode_r)
st.image("https://raw.githubusercontent.com/naaufald/METNUM/main/NA.png")

st.markdown("""melakukan normalisasi""")
kode_r = """
```r
norm_data<- scale(data)
norm_data"""
st.markdown(kode_r)
st.image("https://raw.githubusercontent.com/naaufald/METNUM/main/normal.png")
st.markdown("""
beberapa hal yang dilakukan :
1. memilih fitur : melakukan eliminasi ke fitur yang memiliki korelasi tinggi (dalam kasus ini Tx dan Tn) karena dianggap memiliki informasi yang mirip.
menghapus beberapa kolom lain juga yang memiliki NA tinggi (seperti RR dan station_id yang bertipe chr)
2. menangani missing value : menggunakan nilai median untuk menangani missing value
3. transformasi data yang bukan numerik menjadi numerik
4. melakukan normalisasi menggunakan scale()""")

st.subheader("Fast Fourier Transform")
kode_r="""
```r
time_series <- data$Tavg
time_series_detrended <- time_series - mean(time_series, na.rm = TRUE)
time_series_scaled <- scale(time_series_detrended)
metnum_fft <- fft(time_series_scaled)
magnitude <- Mod(metnum_fft)  # Ambil magnitudo FFT
N <- length(time_series_scaled)
half_N <- floor(N / 2)  # Pastikan ambil bilangan bulat
freq <- seq(0, 0.5, length.out = half_N)  
magnitude <- magnitude[1:half_N]"""
st.markdown(kode_r)
kode_r="""
```r
fft_df <- data.frame(Frekuensi = freq, Magnitudo = magnitude)

ggplot(fft_df, aes(x = Frekuensi, y = Magnitudo)) +
  geom_line(color = "blue") +
  labs(title = "Spektrum Frekuensi FFT",
       x = "Frekuensi",
       y = "Magnitudo") +
  theme_minimal()"""
st.markdown(kode_r)
st.image("https://raw.githubusercontent.com/naaufald/METNUM/main/spektrum.png")

st.subheader("Evaluation and Discussion")
st.markdown("""
berdasarkan hasil evaluasi yang telah dilaksanakan, variabel Tavg tetap dipertahankan dalam model karena menunjukkan hubungan yang signifikan dengan Tn dan Tx.
hal ini dilakukan untuk mengatasi multikolinearitas. dalam memastikan bahwa distribusi data tidak dipengaruhi oleh adanya outlier, nilai NA digantikan oleh median selama
proses transformasi data yang dilakukan. selanjutnya dilakukan proses normalisasi dengan fungsi scale() agar semua variabel memiliki skala yang sama atau seragam.

Pemanfaatan FFT atau Fast Fourier Transformation dimanfaatkan untuk mengenai pola frekuensi data waktu, dimana analisis spektrum yang dihasilkan mengungkapkan bahwa
sebagian besar terfokus pada frekuensi yang rendah sehingga menunjukkan adanya tren yang kuat dengan variasi-variasi minimal.""")
