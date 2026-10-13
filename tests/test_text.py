"""Urdu/English text preparation: numbers, pronunciation rules, script splitting, scoring."""


def test_urdu_numbers(engine):
    u = engine.ur_int
    assert u(0) == "صفر" and u(21) == "اکیس" and u(99) == "ننانوے"
    assert u(100) == "ایک سو"
    assert u(2026) == "دو ہزار چھبیس"
    assert u(150000) == "ایک لاکھ پچاس ہزار"
    assert u(12500000) == "ایک کروڑ پچیس لاکھ"
    assert u(3_000_000_000) == "تین ارب"


def test_numbers_to_words_urdu(engine):
    out = engine.numbers_to_words("سال 2026، یعنی 3.5 فیصد اور 45% لوگ، ۱۲ بجے اور 1,500 روپے۔", "ur")
    assert out == "سال دو ہزار چھبیس، یعنی تین اعشاریہ پانچ فیصد اور پینتالیس فیصد لوگ، بارہ بجے اور ایک ہزار پانچ سو روپے۔"


def test_numbers_to_words_keeps_punctuation_tight(engine):
    assert " ," not in engine.numbers_to_words("In 2026, 45% of people", "en")


def test_arabic_keyboard_letters_normalised(engine):
    assert engine.clean_script_chars("كيا​") == "کیا"


def test_pronunciation_rules_whole_word_and_longest_first(engine):
    engine.save_pron("Primeura = پرائم یورا\nAI = اے آئی\nAI tools = اے آئی ٹولز\nnot a rule")
    assert engine.load_pron() == {"Primeura": "پرائم یورا", "AI": "اے آئی", "AI tools": "اے آئی ٹولز"}
    out = engine.prepare_text("primeura کے AI tools اور MAIN بات، 2 سال", "ur")
    assert out == "پرائم یورا کے اے آئی ٹولز اور MAIN بات، دو سال"
    engine.save_pron("")


def test_split_script_pauses_follow_punctuation(engine):
    parts = engine.split_script("پہلا جملہ ہے۔ دوسرا جملہ، کچھ لمبا ہے؟\n\nنیا پیراگراف شروع۔ آخری بات!")
    assert parts[0] == ("پہلا جملہ ہے۔ دوسرا جملہ، کچھ لمبا ہے؟", engine.PARA_PAUSE)
    assert parts[-1][0].endswith("!") and parts[-1][1] == engine.PAUSE["!"]


def test_split_script_unpacked_for_streaming(engine):
    assert [p[0] for p in engine.split_script("ایک۔ دو۔ تین۔", pack=False)] == ["ایک۔", "دو۔", "تین۔"]


def test_split_script_caps_long_sentences(engine):
    parts = engine.split_script(("یہ ایک بہت لمبا جملہ ہے جس میں کوئی وقفہ نہیں " * 12).strip() + "۔")
    assert len(parts) > 1 and all(len(p) <= 201 for p, _ in parts)


def test_split_script_empty(engine):
    assert engine.split_script("   \n\n ") == []


def test_cer_ignores_punctuation_case_and_diacritics(engine):
    assert engine.cer("اگر چاند آج رات غائب ہو جائے؟", "اگر چاند آج رات غائب ہو جائے") == 0.0
    assert engine.cer("Hello, World!", "hello world") == 0.0
    assert engine.cer("سَلام", "سلام") == 0.0
    assert 0.3 < engine.cer("اگر چاند آج رات غائب ہو جائے", "اگر سورج آج رات غائب ہو") < 0.5


def test_badges(engine):
    b = engine.badge
    assert (b(0.9, 0.01), b(0.6, 0.01), b(0.75, 0.08), b(None, None)) == ("🟢", "🔴", "🟡", "⚪")


def test_batch_split(engine):
    assert engine.split_batch("ایک\n---\nدو\n\n=====\nتین\n---\n") == ["ایک", "دو", "تین"]
