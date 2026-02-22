# yapay-zeka-proje-1
Genetik Algoritma ile Web Sunucusu Ayarları Optimizasyonu

# Odev1 – Genetik Algoritma ile Web Sunucusu Ayarları Optimizasyonu (Senaryo 8)

Bu repoda iki farklı format birlikte sunulmuştur:
- **`Odev1.py`**: Kodların tek dosyada toplandığı Python script sürümü. Dilerseniz algoritmayı buradan çalıştırarak kullanabilirsiniz.
- **`Odev1.ipynb`**: Jupyter Notebook sürümü. Kod hücreleriyle birlikte **açıklamalar**, **çıktılar** ve **yakınsama grafiği** doğrudan aynı dosya üzerinde görüntülenebilir.

> Değerlendirme sürecinde özellikle notebook dosyasının akışı ve açıklayıcılığı inceleneceği için `Odev1.ipynb` dosyası, projenin ana teslim formatıdır.

---

## 1. Problem Tanımı (Senaryo 8)

Bir yazılım şirketi, web sunucusu ayarlarını maksimum performans için optimize etmek istemektedir. Bu optimizasyon probleminde iki karar değişkeni vardır:

- **x₁:** CPU çekirdeği sayısı  
- **x₂:** RAM miktarı (GB)

Amaç, verilen kısıtları ihlal etmeden performans skorunu en yüksek yapan `(x₁, x₂)` çiftini bulmaktır.

### 1.1 Amaç Fonksiyonu (Maksimizasyon)

Performans skoru:

\[
y = 5x_1 + 7x_2 - 0.1x_1^2 - 0.2x_2^2
\]

Bu fonksiyon:
- `5x₁ + 7x₂` kısmıyla CPU ve RAM artışını “ödüllendirir”,
- `-0.1x₁² - 0.2x₂²` kısmıyla çok yüksek değerleri “azalan getiriler / maliyet” gibi düşünerek cezalandırır.

### 1.2 Değişken Aralıkları

- **CPU çekirdeği (x₁):** \[2, 12\] aralığında, ayrıca **x₁ ≥ 4**
- **RAM (x₂):** \[4, 64\] aralığında

Bu nedenle CPU için pratik aralık:
- **x₁ ∈ \[4, 12\]**

### 1.3 Kısıtlar

1. **Kaynak kısıtı:**  
   \[
   x_1 \cdot x_2 \le 512
   \]
   CPU-RAM çarpımına üst sınır koyar (toplam kapasite/limit gibi düşünülebilir).

2. **Minimum CPU kısıtı:**  
   \[
   x_1 \ge 4
   \]

---

## 2. Neden Genetik Algoritma (GA)?

Genetik Algoritma, özellikle:
- arama uzayı büyük olduğunda,
- hedef fonksiyon türevlenebilir olsa bile kısıtlar/ayrık değerler sebebiyle klasik yöntemler zorlaştığında,
- “en iyiye yakın” çözümler için pratik, esnek ve genellenebilir bir yaklaşım gerektiğinde

sık kullanılan bir optimizasyon yöntemidir.

Bu projede değişkenler pratikte **ayrık/tamsayı** olduğundan (CPU çekirdeği sayısı ve RAM GB),
GA yaklaşımı doğal bir seçimdir.

---

## 3. Kullanılan Yaklaşım ve Çözüm Tasarımı

Bu bölümde, Senaryo 8’deki problem GA ile nasıl modellendiği adım adım açıklanmıştır.

### 3.1 Kromozom (Çözüm Temsili)

Her çözüm (birey), iki gen içeren bir kromozom olarak modellenmiştir:

- **Birey = [cpu_cekirdek, ram_gb]**  
- Örnek: `[12, 18]`

Bu temsil:
- okunabilir,
- sade,
- sunumda anlatması kolay
olduğu için tercih edilmiştir.

### 3.2 Başlangıç Popülasyonu

Başlangıç popülasyonu oluşturulurken mümkün olduğunca **kısıtları sağlayan bireyler** üretilmiştir. Özellikle:

- CPU: `4..12` aralığından seçilir  
- RAM üst sınırı, çarpım kısıtına göre daraltılır:
  \[
  x_2 \le \left\lfloor \frac{512}{x_1} \right\rfloor
  \]
  ve ayrıca `x₂ ≤ 64` korunur.

Bu yöntem, daha ilk nesilden itibaren geçerli çözümlerle başlamayı sağlar ve GA’nın verimini artırır.

### 3.3 Fitness (Uygunluk) Fonksiyonu ve Ceza Yaklaşımı

Genetik algoritma seçim operatörünü çalıştırabilmek için her bireyin bir “uygunluk” değeri olmalıdır.

Bu projede:

- **Fitness = Amaç Fonksiyonu – Ceza**

Kısıt ihlali olursa fitness değeri ciddi şekilde düşürülür.
Bu sayede geçersiz çözümler popülasyonda dezavantajlı hale gelir.

Not: Başlangıçta geçerli birey üretimi yapılmasına rağmen, çaprazlama/mutasyon sonrası sınır dışı değerler oluşabileceği için ceza mekanizması ek bir güvence olarak kullanılmıştır.

### 3.4 Seçim Operatörü: Rulet Tekerleği Seçimi

Bu projede seçim (parent selection) için **Rulet Tekerleği (Roulette Wheel Selection)** kullanılmıştır.

Mantık:
- Her bireyin seçilme olasılığı fitness değeriyle orantılıdır.
- Fitness yüksekse seçilme şansı artar; fitness düşükse azalır.
- Bu sayede hem iyi çözümler daha çok seçilir hem de tamamen deterministik bir seçim olmadığı için çeşitlilik korunabilir.

**Negatif fitness** olasılığına karşı, rulet olasılıklarını hesaplamadan önce fitness değerleri pozitif aralığa kaydırılmıştır:
- Eğer minimum fitness < 0 ise tüm fitness değerlerine `-min + 1` kadar ekleme yapılır.
Bu işlem yalnızca olasılık hesabı içindir; “göreli kalite” korunur.

### 3.5 Çaprazlama (Crossover): Uniform Crossover

İki ebeveynden iki çocuk üretmek için **Uniform Crossover** kullanılmıştır:
- Her gen, %50 olasılıkla ebeveynlerden birinden seçilir.
- Derste anlatılan yapıya uygun olarak, genlerin bir kısmı birinci bireyden kalan kısmı ise ikinci bireyden alınarak karma bir çaprazlama işlemi gerçekleştirilmiştir.


### 3.6 Mutasyon (Mutation) ve Onarım (Repair)

Mutasyon, popülasyonun çeşitliliğini korur ve algoritmanın yerel optimumlarda takılmasını azaltır.

Bu projede:
- CPU geninde küçük bir değişim: `±1`
- RAM geninde küçük bir değişim: `±1` veya `±2`

Mutasyon sonrasında çözüm kısıt dışına çıkabilir. Bu yüzden **onarım (repair)** uygulanır:
- CPU ve RAM değerleri kendi aralıklarına kırpılır (clamp),
- çarpım kısıtı ihlal ediliyorsa RAM, `512 // cpu` üst sınırına çekilir.

Bu yaklaşım sayesinde:
- Bireyler mümkün olduğunca geçerli kalır,
- GA döngüsü “geçersiz çözümler” yüzünden verimsizleşmez.

### 3.7 Elitizm (Elitism)

Elitizm açıktır. Her jenerasyonda:
- En iyi birey yeni popülasyona **doğrudan aktarılır**.

Bu yöntem, en iyi çözümün mutasyon/çaprazlama gibi işlemlerle kaybolmasını engeller ve yakınsamayı güçlendirir.

---

## 4. Parametreler (GA Ayarları)

Notebook ve script içinde temel GA parametreleri aşağıdaki gibidir (gerekirse değiştirilebilir):

- Popülasyon boyutu: `pop_boyutu = 40`
- Jenerasyon sayısı: `jenerasyon_sayisi = 80`
- Çaprazlama oranı: `caprazlama_orani = 0.9`
- Mutasyon oranı: `mutasyon_orani = 0.2`
- Elitizm: `True`
- Raporlama adımı: `rapor_adim = 10` (her 10 jenerasyonda bir çıktı)

---

## 5. Çıktılar ve Sonuçların Yorumlanması

Algoritma çalıştırıldığında şu çıktılar üretilir:
- Her belirli jenerasyonda “en iyi birey” ve fitness değeri
- Son jenerasyonda en iyi çözüm, amaç fonksiyonu skoru ve kısıt kontrolü
- Yakınsama grafiği:
  - **En iyi fitness**
  - **Ortalama fitness**

Yakınsama grafiğinin temel yorumu:
- En iyi fitness değerinin hızlı yükselip sabitlenmesi, algoritmanın iyi bir çözüme yakınsadığını gösterir.
- Ortalama fitnessın yükselmesi, popülasyonun genel kalitesinin arttığını gösterir.

Ayrıca bulunan çözümün kısıt sağladığı kod içinde doğrulanır:
- `kisit_sagliyor mu?: True`
- `cpu * ram <= 512`

---
Tanımlanan problem, Genetik Algoritma yaklaşımı kullanılarak başarıyla modellenmiş ve kısıtlar altında uygun bir çözüm elde edilmiştir. Bu doğrultuda proje başarıyla tamamlanmıştır.
