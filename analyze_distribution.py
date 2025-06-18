import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Veriyi oku
df = pd.read_csv('probs.csv')
probs = df['probability'].values

print("=== PROBABILITY DAĞILIMI ANALİZİ ===")
print(f"Toplam örnek sayısı: {len(probs)}")
print(f"Minimum değer: {probs.min():.4f}")
print(f"Maximum değer: {probs.max():.4f}")
print(f"Ortalama: {probs.mean():.4f}")
print(f"Medyan: {np.median(probs):.4f}")
print(f"Standart sapma: {probs.std():.4f}")

# Yüzdelik dilimler
percentiles = [10, 25, 50, 75, 90, 95, 99]
print("\n=== YÜZDELİK DİLİMLER ===")
for p in percentiles:
    value = np.percentile(probs, p)
    print(f"{p}%: {value:.4f}")

# Mevcut eşikler
current_low = 0.40
current_high = 0.60

print(f"\n=== MEVCUT EŞİKLER ===")
print(f"Low (< {current_low}): {np.sum(probs < current_low)} örnek ({np.sum(probs < current_low)/len(probs)*100:.1f}%)")
print(f"Medium ({current_low}-{current_high}): {np.sum((probs >= current_low) & (probs < current_high))} örnek ({np.sum((probs >= current_low) & (probs < current_high))/len(probs)*100:.1f}%)")
print(f"High (>= {current_high}): {np.sum(probs >= current_high)} örnek ({np.sum(probs >= current_high)/len(probs)*100:.1f}%)")

# Önerilen eşikler (yüzdelik dilimlere göre)
print(f"\n=== ÖNERİLEN EŞİKLER (YÜZDELİK DİLİMLERE GÖRE) ===")
low_33 = np.percentile(probs, 33)
high_67 = np.percentile(probs, 67)
print(f"33% yüzdelik: {low_33:.4f}")
print(f"67% yüzdelik: {high_67:.4f}")

print(f"\nÖnerilen eşikler:")
print(f"Low (< {low_33:.4f}): {np.sum(probs < low_33)} örnek ({np.sum(probs < low_33)/len(probs)*100:.1f}%)")
print(f"Medium ({low_33:.4f}-{high_67:.4f}): {np.sum((probs >= low_33) & (probs < high_67))} örnek ({np.sum((probs >= low_33) & (probs < high_67))/len(probs)*100:.1f}%)")
print(f"High (>= {high_67:.4f}): {np.sum(probs >= high_67)} örnek ({np.sum(probs >= high_67)/len(probs)*100:.1f}%)")

# Histogram çiz
plt.figure(figsize=(12, 8))

# Ana histogram
plt.subplot(2, 2, 1)
plt.hist(probs, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(current_low, color='orange', linestyle='--', label=f'Mevcut Low ({current_low})')
plt.axvline(current_high, color='red', linestyle='--', label=f'Mevcut High ({current_high})')
plt.axvline(low_33, color='green', linestyle='-', label=f'Önerilen Low ({low_33:.3f})')
plt.axvline(high_67, color='purple', linestyle='-', label=f'Önerilen High ({high_67:.3f})')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.title('Probability Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

# Kümülatif dağılım
plt.subplot(2, 2, 2)
sorted_probs = np.sort(probs)
cumulative = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
plt.plot(sorted_probs, cumulative, linewidth=2)
plt.axvline(current_low, color='orange', linestyle='--', label=f'Mevcut Low ({current_low})')
plt.axvline(current_high, color='red', linestyle='--', label=f'Mevcut High ({current_high})')
plt.axvline(low_33, color='green', linestyle='-', label=f'Önerilen Low ({low_33:.3f})')
plt.axvline(high_67, color='purple', linestyle='-', label=f'Önerilen High ({high_67:.3f})')
plt.xlabel('Probability')
plt.ylabel('Cumulative Probability')
plt.title('Cumulative Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

# Box plot
plt.subplot(2, 2, 3)
plt.boxplot(probs, vert=False)
plt.axvline(current_low, color='orange', linestyle='--', label=f'Mevcut Low ({current_low})')
plt.axvline(current_high, color='red', linestyle='--', label=f'Mevcut High ({current_high})')
plt.axvline(low_33, color='green', linestyle='-', label=f'Önerilen Low ({low_33:.3f})')
plt.axvline(high_67, color='purple', linestyle='-', label=f'Önerilen High ({high_67:.3f})')
plt.xlabel('Probability')
plt.title('Box Plot')
plt.legend()
plt.grid(True, alpha=0.3)

# Risk kategorileri karşılaştırması
plt.subplot(2, 2, 4)
categories = ['Low', 'Medium', 'High']
current_counts = [
    np.sum(probs < current_low),
    np.sum((probs >= current_low) & (probs < current_high)),
    np.sum(probs >= current_high)
]
suggested_counts = [
    np.sum(probs < low_33),
    np.sum((probs >= low_33) & (probs < high_67)),
    np.sum(probs >= high_67)
]

x = np.arange(len(categories))
width = 0.35

plt.bar(x - width/2, current_counts, width, label='Mevcut Eşikler', alpha=0.7)
plt.bar(x + width/2, suggested_counts, width, label='Önerilen Eşikler', alpha=0.7)
plt.xlabel('Risk Kategorisi')
plt.ylabel('Örnek Sayısı')
plt.title('Risk Kategorileri Karşılaştırması')
plt.xticks(x, categories)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('distribution_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n=== SONUÇ ===")
print(f"Mevcut eşikler ({current_low}, {current_high}) dengesiz bir dağılım oluşturuyor.")
print(f"Önerilen eşikler ({low_33:.3f}, {high_67:.3f}) daha dengeli bir dağılım sağlıyor.")
print(f"Analiz sonuçları 'distribution_analysis.png' dosyasına kaydedildi.") 