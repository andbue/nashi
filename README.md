# nashi
Some bits of javascript to transcribe scanned pages using PageXML. [Try it!](https://andbue.github.io/nashi/nashi.html?pagexml=Test.xml)

## How to start
- Put nashi.html in a folder with your PageXML files (containing line segmentation data) and the page images. Serve the folder in a webserver of your choice or simply use the file:// protocol (only supported in Firefox at the moment).

- In the browser, open the interface as .../path/to/nashi.html?pagexml=Test.xml&direction=rtl where Test.xml is one of the PageXML files and rtl (or ltr) indicates the main direction of your text.

- Install the "Andron Scriptor Web" font to use the additional range of characters

## The interface
- Lines without existing text are marked red, lines containing OCR data blue and lines already transcribed are coloured green
### Keyboard shortcuts in the text input area
- Tab/Shift+Tab switches to the next/previous input
- Shift+Enter saves the edits for the current line
- Shift+Alt shows an additional range of characters to select as an alternative to the character next to the cursor. Input one of them using the 
corresponding number
### Global keyboard shortcuts
- Shift+PageUp/PageDown loads the next/previous page if the filenames of your PageXML files contain the number
- Ctrl+S downloads the PageXML file 

## Planned features
- Simple server in flask
- Help, list of shortcuts
- Add fields for comments, user defined shortcuts...
