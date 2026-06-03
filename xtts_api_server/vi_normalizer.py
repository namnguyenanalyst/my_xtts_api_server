import re

def num_to_vi(num_str):
    if not num_str.isdigit():
        return num_str
        
    num = int(num_str)
    if num == 0:
        return "không"
        
    units = ["", "nghìn", "triệu", "tỷ", "nghìn tỷ", "triệu tỷ"]
    digits = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
    
    def read_group_of_3(n, has_hundreds=False):
        if n == 0:
            return ""
        
        hundred = n // 100
        remainder = n % 100
        ten = remainder // 10
        unit = remainder % 10
        
        res = []
        if has_hundreds or hundred > 0:
            res.append(digits[hundred])
            res.append("trăm")
            
        if ten == 0:
            if unit > 0 and (has_hundreds or hundred > 0):
                res.append("lẻ")
        elif ten == 1:
            res.append("mười")
        else:
            res.append(digits[ten])
            res.append("mươi")
            
        if unit == 1 and ten > 1:
            res.append("mốt")
        elif unit == 5 and ten > 0:
            res.append("lăm")
        elif unit == 4 and ten > 1:
            res.append("tư")
        elif unit > 0:
            res.append(digits[unit])
            
        return " ".join(res)
        
    parts = []
    group_idx = 0
    while num > 0:
        group = num % 1000
        num = num // 1000
        if group > 0:
            group_str = read_group_of_3(group, has_hundreds=(num > 0))
            if units[group_idx]:
                parts.append(group_str + " " + units[group_idx])
            else:
                parts.append(group_str)
        group_idx += 1
        
    return " ".join(reversed(parts)).strip()

def normalize_vietnamese_text(text):
    # Các từ viết tắt phổ biến
    abbreviations = {
        r'\bvn\b': 'việt nam',
        r'\btp\b': 'thành phố',
        r'\bubnd\b': 'ủy ban nhân dân',
        r'\bđh\b': 'đại học',
        r'\b(ko|k)\b': 'không',
        r'\b(đc|dc)\b': 'được',
        r'\b(vs)\b': 'với',
        r'\b(tr|trđ)\b': 'triệu đồng',
        r'\b(đ|vnd)\b': 'đồng',
    }
    for pattern, replacement in abbreviations.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
    # Từ điển Anh-Việt G2P (Code-Switching)
    english_g2p = {
        r'\bfacebook\b': 'phây búc',
        r'\b internet\b': 'in tơ nét',
        r'\bsmartphone\b': 'xờ mát phôn',
        r'\biphone\b': 'ai phôn',
        r'\bipad\b': 'ai pát',
        r'\bapple\b': 'áp pồ',
        r'\bgoogle\b': 'gu gồ',
        r'\byoutube\b': 'giu túp',
        r'\btiktok\b': 'tíc tóc',
        r'\bemail\b': 'i meo',
        r'\bwebsite\b': 'oép sai',
        r'\bonline\b': 'on lai',
        r'\boffline\b': 'ọp lai',
        r'\blivestream\b': 'lai chim',
        r'\bvideo\b': 'vi đi ô',
        r'\blaptop\b': 'láp tốp',
        r'\bapp\b': 'áp',
        r'\bwifi\b': 'oai phai',
        r'\bok\b': 'ô kê',
        r'\bcocacola\b': 'co ca co la',
        r'\bAI\b': 'ây ai',
        r'\bChatGPT\b': 'chat di pi ti',
        r'\bZalo\b': 'za lô',
        r'\bGemini\b': 'Che mi nai',
        r'\bClaude\b': 'cờ lau',
    }
    for pattern, replacement in english_g2p.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
    # Dịch ngày tháng: 12/05/2024 -> ngày 12 tháng 05 năm 2024
    def replace_date(match):
        day = match.group(1)
        month = match.group(2)
        year = match.group(3)
        return f"ngày {num_to_vi(day)} tháng {num_to_vi(month)} năm {num_to_vi(year)}"
        
    text = re.sub(r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b', replace_date, text)
    
    # Dịch số sang chữ
    def replace_num(match):
        num_str = match.group(0).replace('.', '').replace(',', '')
        try:
            return num_to_vi(num_str)
        except:
            return match.group(0)
            
    text = re.sub(r'\b\d+([.,]\d+)*\b', replace_num, text)
    
    return text
