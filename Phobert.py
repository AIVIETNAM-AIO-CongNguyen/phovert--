#Chuẩn hóa viết tắt và lọc từ nhạy cảm và phân loại cảm xúc 

# khởi tạo file ứng dụng 

import streamlit as st 
from underthesea import word_tokenize, sentiment # Thư viện NLP xử lí tiếng việt 
from transformers import pipeline # Để gọi mô hình AI PhoBert
from nltk.tokenize import TreebankWordDetokenizer, wordpunct_tokenize  # tách ghép token

MIN_INPUT_LENGTH = 3

# Khai báo cấu hình chức năng chính: lọc từ nhạy cảm và chuẩn hóa viết tắt và phân loại cảm xúc


Toxic_words = [
    "Cái đồ chết tiệt này muốn chết không",
    "oh shit, mày là cái quái gì thế",
    "Thằng chó kia ",
]

Abbreviation_dict = {
    "ko": "không", "k": "không", "khg": "không",
    "ib": "inbox", "pm": "nhắn tin riêng", "km": "khuyến mãi", "G9": "Goodnight",
    "sp": "sản phẩm", "stk": "số tài khoản", "tl": "trả lời",
    "rep": "phản hồi", "hcm": "Thành phố Hồ Chí Minh", "hn": "Hà Nội",
    "đc": "được", "dc": "được", "j": "gì", "cmt": "comment",
}


#Xây dựng các hàm hỗ trợ 
@st.cache_resource
def init_nlp_tools():
    # Nạp công cụ NLP và mô hình PhoBERT
    detokenizer = TreebankWordDetokenizer()
    phobert_classifier = pipeline ("text-classification", model="wonrax/phobert-base-vietnamese-sentiment")
    return detokenizer, phobert_classifier

detokenizer, phobert_classifier = init_nlp_tools()

def process_text_and_filter(text):
    tokens = word_tokenize(text)
    corrected_tokens = []
    has_changed = False   # chưa có thay đổi gì
    has_toxic = False   # chưa có từ nhạy cảm nào

    # kiểm tra từ bậy 
    for token in tokens:
        token_lower = token.lower()
        if token_lower in Toxic_words:
            censored = "*" * len(token)   
            corrected_tokens.append(censored)
            has_toxic = True 
            has_changed = True 

    # kiểm tra viết tắt
        elif token_lower in Abbreviation_dict:
            expanded = Abbreviation_dict[token_lower]
            if token[0].isupper():
             expanded = expanded.capitalize() # hàm capitalize() để viết hoa chữ cái đầu tiên của từ
            corrected_tokens.append(expanded)
            has_changed = True

    # từ bình thường 
        else: 
            corrected_tokens.append(token)

    #ghép lại thành câu chuẩn khoảng trắng        
    final_text = detokenizer.detokenize(corrected_tokens) # hàm detokenize() để ghép các token lại thành câu chuẩn khoảng trắng
    return final_text, has_changed, has_toxic

# Thiết kế giao diện 
st.set_page_config(page_title="Chuẩn hóa và lọc từ nhạy cảm", page_icon=":guardsman:", layout="wide")
st.title("Chuẩn hóa viết tắt và lọc từ nhạy cảm của chatbot Phobert ")
st.caption("Nhập văn bản để chuẩn hóa viết tắt, lọc từ nhạy cảm và phân loại cảm xúc")

user_input = st.text_area(
    "Nhập bình luận của khách hàng cần kiểm duyệt:",
    value="Nhập vào đây đi nhé!",
    height=130
)
if st.button("Bắt đầu phân tích", type="primary"):
    if not user_input.strip():      # strip loại bỏ khoảng trắng dư thừa 
        st.warning("Vui lòng nhập nội dung văn bản để phân tích!")
    else:
        st.write("---")
        st.subheader("Kết quả phân tích từ hệ thống mạng thần kinh:")
        
        # Bước 1 & 2: Chuẩn hóa chữ nghĩa và lọc từ bậy trước để dữ liệu sạch sẽ
        clean_text, changed, toxic = process_text_and_filter(user_input.strip())
        
        # Bước 3: Đưa văn bản sạch qua bộ não PhoBERT AI để đoán cảm xúc
        with st.spinner("PhoBERT AI đang đọc hiểu ngữ cảnh câu..."):
            ai_result = phobert_classifier(clean_text)
            
        # Mô hình wonrax trả về nhãn dạng: [{'label(nhãn)': 'NEGATIVE', 'score': 0.98}] hoặc 'POSITIVE', 'NEUTRAL'
        label = ai_result[0]['label'].upper()
        confidence = ai_result[0]['score'] * 100  # nhân 100 để chuyển sang phần trăm
        
        # --- HIỂN THỊ KẾT QUẢ SANG GIAO DIỆN ĐẸP ---
        st.markdown("Phân loại cảm xúc khách hàng (PhoBERT AI):")
        
        if "POS" in label: # Positive - Tích cực
            st.success(f"**TÍCH CỰC (Positive):** Khách hàng đang hài lòng! (Độ tin cậy: {confidence:.2f}%)")
            st.balloons()
        elif "NEG" in label: # Negative - Tiêu cực
            st.error(f"**TIÊU CỰC (Negative):** Khách hàng đang phàn nàn, tức giận! (Độ tin cậy: {confidence:.2f}%)")
        else: # Neutral - Trung lập
            st.warning(f"**TRUNG LẬP (Neutral):** Bình luận bình thường, không rõ cảm xúc. (Độ tin cậy: {confidence:.2f}%)")
            
        # KHU VỰC VĂN BẢN ĐÃ CHUẨN HÓA
        st.markdown("2. Văn bản sau khi xử lý & kiểm duyệt:")
        st.info(clean_text)
