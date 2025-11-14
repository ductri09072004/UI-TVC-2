#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script phân tích dataset để đánh giá chất lượng"""

import json
import os
from collections import Counter

def analyze_dataset():
    """Phân tích dataset train và dev"""
    
    # Load data
    train_data = []
    with open('data/train.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            train_data.append(json.loads(line))
    
    dev_data = []
    with open('data/dev.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            dev_data.append(json.loads(line))
    
    def analyze_split(data, name):
        """Phân tích một split của dataset"""
        print(f"\n{'='*70}")
        print(f"📊 PHÂN TÍCH {name.upper()} DATASET")
        print(f"{'='*70}")
        
        total = len(data)
        positive = sum(1 for d in data if d.get('is_match', True))
        negative = total - positive
        
        sentences = [d['sentence'] for d in data]
        word_lengths = [len(s.split()) for s in sentences]
        char_lengths = [len(s) for s in sentences]
        
        # Phân tích độ dài
        avg_words = sum(word_lengths) / len(word_lengths) if word_lengths else 0
        avg_chars = sum(char_lengths) / len(char_lengths) if char_lengths else 0
        min_words, max_words = min(word_lengths), max(word_lengths)
        min_chars, max_chars = min(char_lengths), max(char_lengths)
        
        # Phân bố độ dài
        word_length_dist = Counter(word_lengths)
        common_lengths = word_length_dist.most_common(10)
        
        print(f"📈 Tổng số mẫu: {total:,}")
        print(f"✅ Positive (is_match=True): {positive:,} ({positive/total*100:.1f}%)")
        print(f"❌ Negative (is_match=False): {negative:,} ({negative/total*100:.1f}%)")
        print(f"\n📏 Độ dài câu:")
        print(f"   - Trung bình: {avg_words:.1f} từ, {avg_chars:.1f} ký tự")
        print(f"   - Min/Max: {min_words}/{max_words} từ, {min_chars}/{max_chars} ký tự")
        print(f"\n📊 Phân bố độ dài (top 10):")
        for length, count in common_lengths:
            print(f"   - {length} từ: {count:,} mẫu ({count/total*100:.1f}%)")
        
        # Kiểm tra negative samples
        if negative > 0:
            print(f"\n⚠️  Negative samples:")
            neg_samples = [d for d in data if not d.get('is_match', True)]
            neg_sentences = [d['sentence'] for d in neg_samples[:5]]
            for i, sent in enumerate(neg_sentences, 1):
                print(f"   {i}. {sent[:80]}...")
        
        # Kiểm tra các vấn đề tiềm ẩn
        issues = []
        
        # 1. Câu quá ngắn
        very_short = sum(1 for wl in word_lengths if wl < 3)
        if very_short > 0:
            issues.append(f"⚠️  {very_short} câu quá ngắn (<3 từ) - có thể gây khó khăn cho model")
        
        # 2. Câu quá dài
        very_long = sum(1 for wl in word_lengths if wl > 50)
        if very_long > 0:
            issues.append(f"⚠️  {very_long} câu quá dài (>50 từ) - có thể bị cắt bớt")
        
        # 3. Câu trống hoặc chỉ có khoảng trắng
        empty = sum(1 for s in sentences if not s.strip())
        if empty > 0:
            issues.append(f"❌ {empty} câu trống - cần loại bỏ")
        
        # 4. Kiểm tra ký tự đặc biệt
        special_chars = sum(1 for s in sentences if any(c in s for c in ['\\r\\n', '\\n', '\t']))
        if special_chars > 0:
            issues.append(f"⚠️  {special_chars} câu có ký tự đặc biệt (\\r\\n, \\t) - nên normalize")
        
        if issues:
            print(f"\n🔍 Vấn đề tiềm ẩn:")
            for issue in issues:
                print(f"   {issue}")
        else:
            print(f"\n✅ Không phát hiện vấn đề nghiêm trọng")
        
        return {
            'total': total,
            'positive': positive,
            'negative': negative,
            'avg_words': avg_words,
            'avg_chars': avg_chars,
            'min_words': min_words,
            'max_words': max_words
        }
    
    train_stats = analyze_split(train_data, "TRAIN")
    dev_stats = analyze_split(dev_data, "DEV")
    
    # So sánh train vs dev
    print(f"\n{'='*70}")
    print(f"📊 SO SÁNH TRAIN vs DEV")
    print(f"{'='*70}")
    print(f"Tỷ lệ train/dev: {train_stats['total']/dev_stats['total']:.2f}:1")
    print(f"Độ dài trung bình train: {train_stats['avg_words']:.1f} từ")
    print(f"Độ dài trung bình dev: {dev_stats['avg_words']:.1f} từ")
    
    if abs(train_stats['avg_words'] - dev_stats['avg_words']) > 2:
        print(f"⚠️  Cảnh báo: Độ dài câu train và dev khác nhau đáng kể - có thể gây distribution shift")
    
    # Kiểm tra audio files
    print(f"\n{'='*70}")
    print(f"🎵 KIỂM TRA AUDIO FILES")
    print(f"{'='*70}")
    
    audio_dir = 'archive/mp3'
    if not os.path.exists(audio_dir):
        audio_dir = 'archive/waves'
    
    if os.path.exists(audio_dir):
        all_audio_files = set(os.listdir(audio_dir))
        train_audio = set(d['audio'] for d in train_data)
        dev_audio = set(d['audio'] for d in dev_data)
        
        train_missing = train_audio - all_audio_files
        dev_missing = dev_audio - all_audio_files
        
        print(f"📁 Thư mục audio: {audio_dir}")
        print(f"📦 Tổng số file audio: {len(all_audio_files):,}")
        print(f"🎯 Train audio files: {len(train_audio):,}")
        print(f"🎯 Dev audio files: {len(dev_audio):,}")
        
        if train_missing:
            print(f"❌ Train: {len(train_missing)} file audio không tồn tại")
            print(f"   Ví dụ: {list(train_missing)[:3]}")
        
        if dev_missing:
            print(f"❌ Dev: {len(dev_missing)} file audio không tồn tại")
            print(f"   Ví dụ: {list(dev_missing)[:3]}")
        
        if not train_missing and not dev_missing:
            print(f"✅ Tất cả file audio đều tồn tại")
    else:
        print(f"⚠️  Không tìm thấy thư mục audio: {audio_dir}")

if __name__ == "__main__":
    analyze_dataset()

