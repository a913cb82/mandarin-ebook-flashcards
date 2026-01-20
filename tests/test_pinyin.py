from pinyin import convert_pinyin


def test_convert_pinyin():
    assert convert_pinyin("hao3") == "hǎo"
    assert convert_pinyin("nv3") == "nǚ"
    assert convert_pinyin("Lü4") == "Lǜ"
