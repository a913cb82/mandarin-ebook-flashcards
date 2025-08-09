from ebooklib import epub
import os

def create_test_epub(file_path):
    book = epub.EpubBook()

    # set metadata
    book.set_identifier('id123456')
    book.set_title('Test Book')
    book.set_language('zh')

    book.add_author('Author')

    # create chapters
    c1 = epub.EpubHtml(title='第一章', file_name='chap_1.xhtml', lang='zh')
    c1.content=u'<html><head></head><body><h1>第一章</h1><p>这是第一章。</p></body></html>'

    c2 = epub.EpubHtml(title='第二章', file_name='chap_2.xhtml', lang='zh')
    c2.content=u'<html><head></head><body><h1>第二章</h1><p>这是第二章。</p></body></html>'

    c3 = epub.EpubHtml(title='第三章', file_name='chap_3.xhtml', lang='zh')
    c3.content=u'<html><head></head><body><h1>第三章</h1><p>这是第三章。</p></body></html>'

    # add chapters to the book
    book.add_item(c1)
    book.add_item(c2)
    book.add_item(c3)

    # add default NCX and Nav file
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    # define the spine
    book.spine = ['nav', c1, c2, c3]

    # write to the file
    epub.write_epub(file_path, book, {})

if __name__ == '__main__':
    # Get the absolute path to the project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.join(project_dir, 'src/tests')
    if not os.path.exists(tests_dir):
        os.makedirs(tests_dir)
    create_test_epub(os.path.join(tests_dir, 'test_book.epub'))
