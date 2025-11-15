#!/usr/bin/env python3
"""
Script để loại bỏ các dòng trùng lặp trong file text/CSV.
Giữ lại thứ tự ban đầu và chỉ giữ lại lần xuất hiện đầu tiên của mỗi dòng.
"""

import sys
import os

def remove_duplicates(input_file: str, output_file: str = None):
    """
    Loại bỏ các dòng trùng lặp trong file text/CSV.
    
    Args:
        input_file: Đường dẫn đến file đầu vào (.txt, .csv, ...) hoặc "-" để đọc từ stdin
        output_file: Đường dẫn đến file đầu ra (nếu None, sẽ ghi đè file đầu vào)
    """
    lines = None
    
    # Xử lý đọc từ stdin
    if input_file == "-" or input_file == "/dev/stdin":
        print("📖 Đang đọc từ stdin...")
        try:
            lines = sys.stdin.readlines()
            if not lines:
                print("⚠ Không có dữ liệu từ stdin!")
                return False
        except Exception as e:
            print(f"❌ Lỗi khi đọc từ stdin: {e}")
            return False
    else:
        # Đọc từ file
        if not os.path.exists(input_file):
            print(f"❌ Lỗi: File không tồn tại: {input_file}")
            return False
        
        # Kiểm tra kích thước file
        file_size = os.path.getsize(input_file)
        if file_size == 0:
            print("⚠ CẢNH BÁO: File rỗng hoặc chưa được lưu!")
            print("   Vui lòng lưu file trong editor trước khi chạy script.")
            print("   Hoặc sử dụng: Get-Content data_ban.txt | python remove_duplicates.py -")
            return False
        
        print(f"📖 Đang đọc file: {input_file} ({file_size} bytes)...")
        
        # Đọc tất cả các dòng với nhiều encoding thử
        encodings = ['utf-8', 'utf-8-sig', 'cp1258', 'latin-1']
        
        for encoding in encodings:
            try:
                with open(input_file, 'r', encoding=encoding) as f:
                    lines = f.readlines()
                print(f"✓ Đọc file thành công với encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"❌ Lỗi khi đọc file với encoding {encoding}: {e}")
                continue
        
        if lines is None:
            print("❌ Không thể đọc file với các encoding đã thử!")
            return False
        
        if not lines:
            print("⚠ File không có nội dung!")
            return False
    
    print(f"📊 Tổng số dòng đọc được: {len(lines)}")
    
    # Loại bỏ trùng lặp, giữ lại thứ tự ban đầu
    seen = set()
    unique_lines = []
    duplicate_positions = []  # Lưu vị trí các dòng trùng
    
    print("🔄 Đang xử lý loại bỏ trùng lặp...")
    
    for idx, line in enumerate(lines, start=1):
        # Chuẩn hóa: loại bỏ khoảng trắng đầu/cuối và chuyển sang lowercase để so sánh
        # Loại bỏ khoảng trắng thừa ở giữa
        normalized = ' '.join(line.strip().split()).lower()
        
        # Nếu dòng trống sau khi normalize, bỏ qua
        if not normalized:
            continue
        
        # Nếu chưa thấy, thêm vào danh sách
        if normalized not in seen:
            seen.add(normalized)
            # Giữ nguyên dòng gốc (chỉ strip newline, giữ khoảng trắng và chữ hoa/thường)
            original_line = line.rstrip('\n\r')
            unique_lines.append(original_line)
        else:
            # Ghi nhận vị trí trùng lặp
            duplicate_positions.append((idx, line.strip()))
    
    # Xác định file output
    if output_file is None:
        output_file = input_file
    
    # Ghi lại file
    print(f"💾 Đang ghi file: {output_file}...")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in unique_lines:
                f.write(line + '\n')
    except Exception as e:
        print(f"❌ Lỗi khi ghi file: {e}")
        return False
    
    # Thống kê
    original_count = len([l for l in lines if l.strip()])
    unique_count = len(unique_lines)
    removed_count = original_count - unique_count
    
    print("\n" + "="*50)
    print("✓ Đã xử lý xong!")
    print("="*50)
    print(f"  📝 Tổng số dòng ban đầu: {original_count}")
    print(f"  ✅ Số dòng sau khi loại bỏ trùng: {unique_count}")
    print(f"  🗑️  Số dòng đã loại bỏ: {removed_count}")
    print(f"  📁 File đã được lưu: {output_file}")
    
    # Hiển thị một số ví dụ về dòng trùng (nếu có)
    if duplicate_positions:
        print(f"\n  📋 Ví dụ các dòng trùng (hiển thị 10 dòng đầu tiên):")
        for pos, content in duplicate_positions[:10]:
            print(f"    - Dòng {pos}: {content}")
        if len(duplicate_positions) > 10:
            print(f"    ... và {len(duplicate_positions) - 10} dòng trùng khác")
    else:
        print("\n  ℹ️  Không có dòng trùng lặp nào!")
    
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("="*50)
        print("Script loại bỏ dòng trùng lặp trong file text/CSV")
        print("="*50)
        print("\nCách sử dụng:")
        print("  python remove_duplicates.py <input_file> [output_file]")
        print("\nVí dụ:")
        print("  python remove_duplicates.py data_ban.txt")
        print("  python remove_duplicates.py data_ban.csv")
        print("  python remove_duplicates.py data_ban.txt data_ban_cleaned.txt")
        print("\nLưu ý:")
        print("  - Nếu không chỉ định output_file, file gốc sẽ bị ghi đè")
        print("  - Script sẽ tự động thử nhiều encoding để đọc file")
        print("  - So sánh không phân biệt hoa/thường và khoảng trắng thừa")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = remove_duplicates(input_file, output_file)
    sys.exit(0 if success else 1)

