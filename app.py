import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import random
import os

# ============================================================
# 1. إعدادات الصفحة والقاموس (Recommendations)
# ============================================================
st.set_page_config(page_title="EmotiScan-AI", layout="centered", page_icon="🧠")

# القاموس الكامل بناءً على الصور والأكواد التي قدمتها
RECOMMENDATIONS = {
    'happy': {
        'arabic_name': 'سعيد',
        'color': '#f1c40f',
        'quotes': [
            'السعادة ليست وجهة بل طريقة في السفر.',
            'ابتسامتك تضيء العالم من حولك، استمر!',
            'أجمل اللحظات هي تلك التي تشعر فيها أن الحياة تستحق.',
            'السعادة تضاعف حين تُشارك.',
            'كن سعيداً الآن، هذه اللحظة لن تعود.'
        ],
        'tips': ['شارك سعادتك مع من تحب اليوم.', 'هذا وقت رائع لبدء مشروع جديد أو هدف طال انتظاره.', 'اكتب شيئاً تشعر بالامتنان له الآن.']
    },
    'sad': {
        'arabic_name': 'حزين',
        'color': '#3498db',
        'quotes': ['بعد كل عاصفة تشرق الشمس، صبراً.', 'الحزن يمر كما مرّت كل الأشياء، لن يدوم.', 'أحياناً الدموع هي أكثر صور القوة صدقاً.', 'حتى في أحلك اللحظات، أنت لست وحدك.', 'الجرح الذي لا يقتلك يجعلك أقوى.'],
        'tips': ['تحدث مع شخص تثق به عما تشعر به.', 'خذ نفساً عميقاً، وامشِ قليلاً في الهواء الطلق.', 'دلّل نفسك اليوم، أنت تستحق ذلك.']
    },
    'angry': {
        'arabic_name': 'غاضب',
        'color': '#e74c3c',
        'quotes': ['الغضب حرف واحد من الجنون، تذكر ذلك.', 'لا تتخذ قراراً وأنت غاضب، ولا وعداً وأنت سعيد.', 'القوة الحقيقية في السيطرة على النفس لا على الآخرين.', 'تنفس... ثم تكلم. الهدوء يحل ما يعجز عنه الغضب.', 'الغضب عقاب تفرضه على نفسك بسبب خطأ الآخرين.'],
        'tips': ['خذ 10 أنفاس عميقة قبل أي رد فعل.', 'اكتب ما تشعر به على ورقة ثم امزقها.', 'مارس رياضة خفيفة لتفريغ الطاقة السلبية.']
    },
    'fearful': {
        'arabic_name': 'خائف',
        'color': '#9b59b6',
        'quotes': ['الشجاعة ليست غياب الخوف، بل المضي رغمه.', 'كل ما تريده يقع خارج منطقة الراحة.', 'الخوف يطرق الباب، الإيمان يفتحه فيجد لا أحد.', 'أكثر الأشياء التي خشيتها لم تحدث أبداً.', 'افعلها وأنت خائف، هذا هو المعنى الحقيقي للشجاعة.'],
        'tips': ['حدد خوفك بوضوح، المجهول أصعب دائماً من الحقيقة.', 'تذكر موقفاً تغلبت فيه على خوف سابق.', 'تحدث مع نفسك بلطف كما تفعل مع صديق مقرب.']
    },
    'neutral': {
        'arabic_name': 'محايد',
        'color': '#95a5a6',
        'quotes': ['الهدوء قوة، لا تستهن به.', 'في الثبات حكمة لا يراها المتسرعون.', 'لحظة التوازن هي أفضل لحظة للتفكير والتخطيط.', 'الاتزان هو فن الحياة في أبهى صوره.', 'كل يوم هادئ هو يوم تبنى فيه نفسك بصمت.'],
        'tips': ['استغل هدوءك الآن للتخطيط لهدف مهم.', 'اقرأ شيئاً يثري عقلك أو تعلم مهارة جديدة.', 'تواصل مع شخص لم تكلمه منذ فترة.']
    },
    'disgusted': {
        'arabic_name': 'مشمئز',
        'color': '#27ae60',
        'quotes': ['ما لا يعجبك يمكنك تغييره، وما لا تستطيع تغييره يمكنك تجاوزه.', 'رد فعلك تجاه الأشياء يكشف شخصيتك أكثر مما تكشفه تلك الأشياء.', 'صفّ عقلك، تصفو الرؤية.', 'لا تسمح لما تكرهه أن يسرق سلامك الداخلي.', 'ابتعد عما يؤذيك، هذا حكمة لا ضعف.'],
        'tips': ['ابتعد مؤقتاً عن مصدر الانزعاج وأعطِ نفسك مساحة.', 'ركز على ما يمكنك التحكم فيه فعلاً.', 'اشرب ماء وتنفس بعمق، الجسم يؤثر على المزاج.']
    },
    'surprised': {
        'arabic_name': 'مندهش',
        'color': '#e67e22',
        'quotes': ['الدهشة بداية كل اكتشاف عظيم.', 'الحياة مليئة بالمفاجآت، وهذا ما يجعلها مثيرة.', 'كن مستعداً دائماً، فالحياة تحب أن تفاجئنا.', 'أجمل اللحظات هي التي لم تتوقعها.', 'الانبهار بالحياة نعمة، احتفظ بها.'],
        'tips': ['خذ لحظة لاستيعاب ما فاجأك قبل رد الفعل.', 'المفاجآت - سواء كانت جيدة أو سيئة - تحمل دروساً.', 'شارك ما فاجأك مع شخص قريب منك.']
    }
}

# ============================================================
# 2. تحميل الموارد (Model & Cascade)
# ============================================================
IMG_SIZE = 48
CLASS_LABELS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

@st.cache_resource
def load_model():
    # تأكدي من وجود هذا الملف في الفولدر
    model = tf.keras.models.load_model('emotion_model_final.h5')
    # تحميل الـ Cascade لاكتشاف الوجوه
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return model, face_cascade

best_model, face_cascade = load_model()

# ============================================================
# 3. الدوال البرمجية (Logic)
# ============================================================
def get_recommendation(emotion_label):
    label = emotion_label.lower()
    if label not in RECOMMENDATIONS:
        return None
    data = RECOMMENDATIONS[label]
    return {
        'emotion_en': emotion_label,
        'emotion_ar': data['arabic_name'],
        'color': data['color'],
        'quote': random.choice(data['quotes']),
        'tip': random.choice(data['tips'])
    }

def analyze_emotion(image):
    img_array = np.array(image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_disp = img_array.copy()

    faces = face_cascade.detectMultiScale(img_gray, 1.1, 5, minSize=(30, 30))

    if len(faces) == 0:
        return None, "لم يتم العثور على وجه", "", "", "#fff"

    # معالجة أكبر وجه فقط
    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    cv2.rectangle(img_disp, (x, y), (x+w, y+h), (52, 235, 152), 3)

    face_roi = img_gray[y:y+h, x:x+w]
    face_roi = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
    face_input = face_roi.astype('float32') / 255.0
    face_input = np.expand_dims(face_input, axis=[0, -1])

    predictions = best_model.predict(face_input, verbose=0)[0]
    emotion_label = CLASS_LABELS[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    rec = get_recommendation(emotion_label)
    emotion_text = f"{rec['emotion_ar']} ({confidence:.1f}%)"
    
    return img_disp, emotion_text, rec['quote'], rec['tip'], rec['color']

# ============================================================
# 4. واجهة Streamlit (UI)
# ============================================================
st.markdown("""
    <style>
    .main { direction: rtl; text-align: right; }
    .stAlert { direction: rtl; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 EmotiScan-AI")
st.subheader("نظام كشف المشاعر والتوصيات الذكي")

# القائمة الجانبية (Sidebar) لفريق العمل
with st.sidebar:
    st.header("فريق العمل")
    st.write("1. الاسم الأول")
    st.write("2. الاسم الثاني")
    st.write("3. الاسم الثالث")
    st.write("4. الاسم الرابع")
    st.write("5. الاسم الخامس")
    st.divider()
    st.info("مشروع تحليل المشاعر باستخدام FER2013")

uploaded_file = st.file_uploader("ارفع صورة للتحليل...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="الصورة الأصلية", use_container_width=True)
    
    with st.spinner('جاري تحليل المشاعر...'):
        res_img, emotion, quote, tip, color = analyze_emotion(image)
        
    if res_img is not None:
        with col2:
            st.image(res_img, caption="نتيجة الكشف", use_container_width=True)
        
        st.divider()
        # عرض النتائج بشكل جمالي باستخدام الألوان الخاصة بكل حالة
        st.markdown(f"<h2 style='text-align: center; color: {color};'>{emotion}</h2>", unsafe_allow_html=True)
        
        st.success(f"**الاقتباس:** {quote}")
        st.info(f"**نصيحة اليوم:** {tip}")
    else:
        st.error(emotion)

st.divider()
st.caption("EmotiScan-AI — Web-based Mobile Application 2026")
