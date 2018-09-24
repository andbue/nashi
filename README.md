# nashi (nasḫī)
Some bits of javascript to transcribe scanned pages using PageXML. Both ltr and rtl languages are supported. [Try it!](https://andbue.github.io/nashi/nashi.html?pagexml=Test.xml)
But wait, there's more: download now and get a complete webapp written in Python/Flask that handles import and export of your scanned pages to and from [LAREX](https://github.com/chreul/LAREX) for semi-automatic layout analysis, does the line segmentation for you (via [kraken](http://kraken.re/)) and saves your precious PageXML in a database. All you've got to do is follow the instructions below and help me implement all the missing features... OCR training and recognition is currently not included because of our webhost's limited capacity.

## Instructions for nashi.html
- Put nashi.html in a folder with (or some folder above) your PageXML files (containing line segmentation data) and the page images. Serve the folder in a webserver of your choice or simply use the file:// protocol (only supported in Firefox at the moment).
- In the browser, open the interface as .../path/to/nashi.html?pagexml=Test.xml&direction=rtl where Test.xml (or subfolder/Test.xml) is one of the PageXML files and rtl (or ltr) indicates the main direction of your text.
- Install the "Andron Scriptor Web" font to use the additional range of characters.

### The interface
- Lines without existing text are marked red, lines containing OCR data blue and lines already transcribed are coloured green.
### Keyboard shortcuts in the text input area
- Tab/Shift+Tab switches to the next/previous input.
- Shift+Enter saves the edits for the current line.
- Shift+Insert shows an additional range of characters to select as an alternative to the character next to the cursor. Input one of them using the corresponding number while holding Insert.
- Shift+ArrowDown opens a new comment field (Shift+ArrowUp switches back to the transcription line).
### Global keyboard shortcuts
- Ctrl+Space Zooms in to line width
- Shift+PageUp/PageDown loads the next/previous page if the filenames of your PageXML files contain the number.
- Ctrl+Shift+ArrowLeft/ArrowRight changes orientation and input direction to ltr/rtl.
- Ctrl+S downloads the PageXML file.
- Ctrl+E enters or exits polygon edit mode.
### Edit mode
- Click on line area to activate point handles. Points can be moved around using, new points can be created by drawing the borders between existing points.
- If points or lines are active, they can be deleted using the "Delete"-key.
- Hold Shift-key and draw to select multiple points
- New text lines can be created by clicking inside an existing text region and drawing a rectangle. New lines are always added at the end of the region.

## Instructions for the server
- Install [redis](https://redis.io/). The app uses
[celery](http://www.celeryproject.org/) as a task queue for line segmentation jobs (and probably OCR jobs in the future).
- Install [LAREX](https://github.com/chreul/LAREX) for semi-automatic layout analysis.
- Install the server from this repository or from pypi:
```bash
pip install nashi
```
- Create a config.py file. For more options see the file default\_settings.py. If you want the app to send emails to users, change the mail settings there. Here is just a minimal example:
```python
BOOKS_DIR = "/home/username/books/"
LAREX_DIR = "/home/username/larex_books/"

```
- Set an environment variable containing your [database url](http://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls). If you don't, nashi will create a sqlite database called "test.db" in your working directory.
```bash
export DATABASE_URL="mysql+pymysql://user:pw@localhost/mydb?charset=utf8"
```
- Create the database tables (and users, if needed) from a python prompt. Login is disabled in the default config file.
```python
from nashi import user_datastore
from nashi.database import db_session, init_db
init_db()
user_datastore.create_user(email="me@myserver.de.vu", password="secret")
db_session.commit()
```
- Run the celery worker:
```bash
export NASHI_SETTINGS=/home/user/path/to/config.py
celery -A nashi.celery worker --loglevel=info
```
- Run the app, don't forget to export your DATABASE\_URl again if you're using a new terminal:
```bash
export FLASK_APP=nashi
export NASHI_SETTINGS=/home/user/path/to/config.py
flask run
```
- Open [localhost:5000](http://localhost:5000), log in, update your books list via "Edit, Refresh".

## Planned features
- Sorting of lines
- Reading order
- Creation and correction of regions
- API for external OCR service
- Advanced text editing capabilities
- Help, examples, and documentation
- Artificial general intelligence that writes the code for me
