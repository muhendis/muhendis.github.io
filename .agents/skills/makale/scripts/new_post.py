#!/usr/bin/env python3
"""
Yeni bir iki dilli makale çifti (TR + EN) iskeleti kuran scaffolding aracı.
Kullanım:
    python3 new_post.py <tr_slug> <en_slug> "Türkçe Başlık" "English Title"
"""
import sys
import json
import datetime
from pathlib import Path

def main():
    if len(sys.argv) < 5:
        print("Kullanım: python3 new_post.py <tr_slug> <en_slug> \"Türkçe Başlık\" \"English Title\"")
        print("Örnek: python3 new_post.py flash-attention-derinlemesine inside-flash-attention \"FlashAttention Derinlemesine\" \"Inside FlashAttention\"")
        sys.exit(1)

    tr_slug = sys.argv[1].strip()
    en_slug = sys.argv[2].strip()
    tr_title = sys.argv[3].strip()
    en_title = sys.argv[4].strip()

    root = Path(__file__).resolve().parents[4]
    today = datetime.date.today().isoformat()

    tr_md_path = root / f"tr/blog/posts/{tr_slug}.md"
    en_md_path = root / f"en/blog/posts/{en_slug}.md"

    if tr_md_path.exists():
        print(f"Hata: {tr_md_path} zaten mevcut!")
        sys.exit(1)
    if en_md_path.exists():
        print(f"Hata: {en_md_path} zaten mevcut!")
        sys.exit(1)

    # İskelet metinler
    tr_content = f"""[Giriş paragrafı 1: Okurun içinde bulunduğu somut sahne veya mühendislik paradoksuyla başlayın. Okura ödev vermeyin.]

[Giriş paragrafı 2: Gerilimi ölçülebilir bir kaynakla bağlayın (VRAM, gecikme, FLOP, parametre payı).]

Bu yazı, konunun sezgisel temelini, matematiksel ayrışma gerekçesini, elle yapılan küçük boyutlu sayısal hesap yürüyüşünü, eğitim ve çıkarım arasındaki temel donanım asimetrisini ve PyTorch doğrulamasını bütüncül bir dille ortaya koyuyor.

**Bu yazıda**

- [1. En yalın hâliyle temel sezgi](#1-en-yalın-hâliyle-temel-sezgi)
- [2. Matematiksel model ve semboller](#2-matematiksel-model-ve-semboller)
- [3. Adım adım elle sayısal hesap](#3-adım-adım-elle-sayısal-hesap)
- [4. İki mühendislik gözü: Eğitim ve çıkarım](#4-i̇ki-mühendislik-gözü-eğitim-ve-çıkarım)
- [5. PyTorch ile doğrulama](#5-pytorch-ile-doğrulama)
- [Bütün hikâye altı satırda](#bütün-hikâye-altı-satırda)
- [Terimler sözlüğü](#terimler-sözlüğü)
- [Daha derine inmek için](#daha-derine-inmek-için)

---

## 1. En yalın hâliyle temel sezgi

Kavramı zihinde somutlaştıran bir analoji veya mühendislik sahnesi.

```mermaid
flowchart TD
    A["Girdi"] --> B["İşlem"]
    B --> C["Çıktı"]
```

---

## 2. Matematiksel model ve semboller

$$y = f(x)$$

Şöyle okuyun: $x$ girdi tensörü, $y$ ise dönüştürülmüş çıktıdır.

> **Terim Adı** = tek cümlelik teknik tanım.

---

## 3. Adım adım elle sayısal hesap

```text
Oyuncak Boyut (d = 4):
Adım 1: ...
Adım 2: ...
```

---

## 4. İki mühendislik gözü: Eğitim ve çıkarım

| Boyut | Eğitim Tarafı (Training) | Çıkarım Tarafı (Inference) |
| :--- | :--- | :--- |
| **Bellek** | Ağırlık + Gradyan + AdamW (16B) | Ağırlık + KV Cache |
| **Hesap Kısıtı** | Compute-bound | Memory-bandwidth-bound |
| **Riskler** | Loss spike, iletişim gecikmesi | TTFT, TPOT, bellek taşması |

---

## 5. PyTorch ile doğrulama

```python
import torch

# Doğrulama kodu
```

Konsol çıktısı:

```text
# Çıktı
```

---

## Bütün hikâye altı satırda

- 1. Madde
- 2. Madde
- 3. Madde
- 4. Madde
- 5. Madde
- 6. Madde

---

## Terimler sözlüğü

- **Terim** — tanım + sezgi veya sonuç cümlesi.

---

## Daha derine inmek için

- [arXiv Makalesi]
- Bu blogda: [İlgili Makale](post.html?slug=...)
"""

    en_content = f"""[Opening paragraph 1: Start with the concrete engineering scene or paradox the reader is experiencing. Do not assign homework.]

[Opening paragraph 2: Tie the tension to a measurable resource (VRAM, latency, FLOPs, parameter share).]

This article walks through the intuitive foundation, the mathematical formalism, step-by-step manual tensor calculations on a toy example, the fundamental hardware asymmetry between training and inference, and runnable PyTorch verification.

**In this article**

- [1. The core intuition in plain terms](#1-the-core-intuition-in-plain-terms)
- [2. Mathematical model and symbols](#2-mathematical-model-and-symbols)
- [3. Step-by-step manual tensor walk](#3-step-by-step-manual-tensor-walk)
- [4. The two engineering lenses: Training vs inference](#4-the-two-engineering-lenses-training-vs-inference)
- [5. PyTorch verification](#5-pytorch-verification)
- [The whole story in six lines](#the-whole-story-in-six-lines)
- [Glossary](#glossary)
- [Going deeper](#going-deeper)

---

## 1. The core intuition in plain terms

A tangible analogy or engineering reality that grounds the concept.

```mermaid
flowchart TD
    A["Input"] --> B["Process"]
    B --> C["Output"]
```

---

## 2. Mathematical model and symbols

$$y = f(x)$$

Read it as: $x$ is the input tensor, and $y$ is the transformed output representation.

> **Term Name** = single-sentence technical definition.

---

## 3. Step-by-step manual tensor walk

```text
Toy Dimension (d = 4):
Step 1: ...
Step 2: ...
```

---

## 4. The two engineering lenses: Training vs inference

| Dimension | Training Side | Inference Side |
| :--- | :--- | :--- |
| **Memory** | Weights + Gradients + AdamW (16B) | Weights + KV Cache |
| **Bottleneck** | Compute-bound | Memory-bandwidth-bound |
| **Risks** | Loss spikes, communication overhead | TTFT, TPOT, OOM |

---

## 5. PyTorch verification

```python
import torch

# Verification code
```

Console output:

```text
# Output
```

---

## The whole story in six lines

- Point 1
- Point 2
- Point 3
- Point 4
- Point 5
- Point 6

---

## Glossary

- **Term** — definition + intuition or consequence sentence.

---

## Going deeper

- [arXiv Paper]
- On this blog: [Related Post](post.html?slug=...)
"""

    tr_md_path.write_text(tr_content, encoding='utf-8')
    en_md_path.write_text(en_content, encoding='utf-8')
    print(f"Oluşturuldu: {tr_md_path}")
    print(f"Oluşturuldu: {en_md_path}")

    # posts.json dosyalarına draft entry ekleme
    for p_file, slug, title, trans in [
        (root / "tr/blog/posts.json", tr_slug, tr_title, en_slug),
        (root / "en/blog/posts.json", en_slug, en_title, tr_slug)
    ]:
        if p_file.exists():
            data = json.loads(p_file.read_text(encoding='utf-8'))
            entry = {
                "slug": slug,
                "title": title,
                "date": today,
                "summary": "Ne: ... Neden önemli: ... Nasıl: ... Nerede: ... Kim için: ... Ne zaman: ...",
                "tags": ["llm", "mimari"],
                "readingMinutes": 10,
                "translationSlug": trans,
                "draft": True
            }
            data.setdefault('posts', []).insert(0, entry)
            data['updated'] = today
            p_file.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
            print(f"Taslak entry eklendi: {p_file}")

    print("\nYeni makale çifti başarıyla oluşturuldu! Şimdi taslakları doldurabilirsiniz.")

if __name__ == '__main__':
    main()
