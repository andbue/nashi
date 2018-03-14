# nashi (nasḫī)
Some bits of javascript to transcribe scanned pages using PageXML. Both ltr and rtl languages are supported. [Try it!](https://andbue.github.io/nashi/nashi.html?pagexml=Test.xml)

## Instructions
- Put nashi.html in a folder with (or some folder above) your PageXML files (containing line segmentation data) and the page images. Serve the folder in a webserver of your choice or simply use the file:// protocol (only supported in Firefox at the moment).
- In the browser, open the interface as .../path/to/nashi.html?pagexml=Test.xml&direction=rtl where Test.xml (or subfolder/Test.xml) is one of the PageXML files and rtl (or ltr) indicates the main direction of your text.
- Install the "Andron Scriptor Web" font to use the additional range of characters.

### The interface
- Lines without existing text are marked red, lines containing OCR data blue and lines already transcribed are coloured green.
### Keyboard shortcuts in the text input area
- Tab/Shift+Tab switches to the next/previous input.
- Shift+Enter saves the edits for the current line.
- Shift+Insert shows an additional range of characters to select as an alternative to the character next to the cursor. Input one of them using the corresponding number. 
- Shift+ArrowDown opens a new comment field (Shift+ArrowUp switches back to the transcription line).
### Global keyboard shortcuts
- Ctrl+Shift Zooms in to line width
- Shift+PageUp/PageDown loads the next/previous page if the filenames of your PageXML files contain the number.
- Ctrl+Shift+ArrowLeft/ArrowRight changes orientation and input direction to ltr/rtl.
- Ctrl+S downloads the PageXML file.
- Ctrl+E enters or exits polygon edit mode.
### Edit mode
- Click on line area to activate point handles. Points can be moved around using, new points can be created by drawing the borders between existing points.
- If points or lines are active, they can be deleted using the "Delete"-key.
- New text lines can be created by clicking inside an existing text region and drawing a rectangle. New lines are always added at the end of the region.

## Planned features
- Sorting of lines
- Reading order
- Simple server in flask
- Help (list of shortcuts)
