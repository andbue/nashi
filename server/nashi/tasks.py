from nashi import app, make_celery
from nashi.import_from_larex import add_page
from nashi.models import Book
import time

celery = make_celery(app)


@celery.task(bind=True)
def lareximport(self, bookname, pages=[]):
    book = Book.query.filter_by(name=bookname).first()
    message = ''
    lines = 0
    for n, xmlfile in enumerate(pages):
        print(xmlfile)
        message = "Importing page {} of {}.".format(n+1, len(pages))
        self.update_state(state='PROGRESS',
                          meta={'current': n+1, 'total': len(pages),
                                'status': message})
        lines_added = add_page(book, xmlfile, commit=True)
        lines += lines_added
    return {'current': n+1, 'total': len(pages), 'status': 'Import completed!',
            'result': lines}
