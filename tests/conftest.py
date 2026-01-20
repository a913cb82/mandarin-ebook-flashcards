import pytest
from ebooklib import epub


def create_test_epub(file_path):
    book = epub.EpubBook()
    book.set_identifier("id123456")
    book.set_title("Test Book")
    book.set_language("zh")
    book.add_author("Author")

    c1 = epub.EpubHtml(title="第一章", file_name="chap_1.xhtml", lang="zh")
    c1.content = "<html><body><h1>第一章</h1><p>这是第一章。</p></body></html>"
    c2 = epub.EpubHtml(title="第二章", file_name="chap_2.xhtml", lang="zh")
    c2.content = "<html><body><h1>第二章</h1><p>这是第二章。</p></body></html>"
    c3 = epub.EpubHtml(title="第三章", file_name="chap_3.xhtml", lang="zh")
    c3.content = "<html><body><h1>第三章</h1><p>这是第三章。</p></body></html>"

    for c in [c1, c2, c3]:
        book.add_item(c)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", c1, c2, c3]
    epub.write_epub(file_path, book, {})


@pytest.fixture
def test_book_path(tmp_path):
    path = tmp_path / "test_book.epub"
    create_test_epub(str(path))
    return str(path)
