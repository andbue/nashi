
['preserveAspectRatio', 'viewBox', "refX", "refY", "markerWidth", "markerHeight", "markerUnits"].forEach(function(k) {
  // jQuery converts the attribute name to lowercase before
  // looking for the hook.
  $.attrHooks[k.toLowerCase()] = {
    set: function(el, value) {
      if (value) {
        el.setAttribute(k, value);
      } else {
        el.removeAttribute(k, value);
      }
      return true;
    },
    get: function(el) {
      return el.getAttribute(k);
    },
  };
});


let defaultSettings = {
	vKeyboard: {
  "a": ["ä", "ā", "æ", "aͩ", "aͦ", "ȧ", "ª"],
  "b": ["b᷎",	"b", "ƀ", "b̄", "b̅"],
  "c": ["ç", "c̄", "cͣ", "⁋", "¶", "ꝯ", "ͨ"],
  "d": ["d᷎", "", "dͭ", "đ"],
  "e": ["ē", "ę", "eͥ", "ꝫ"],
  "g": ["ḡ", "g̊", "gͥ"],
  "h": ["h̊", "h̄", "ḣ", "h̃"],
  "i": ["ī", "iͥ", "ͥ"],
  "l": ["ł", "ꝉ"],
  "m": ["", "º"],
  "n": [""],
  "o": ["ō", "œ", "º", "ͦ"],
  "p": ["", "p̃",	"ꝑ", "ꝑᷓ", "ꝓ", "ꝑͣ"],
  "q": ["qᷓ", "q̄", "qͥ", "ꝗ", "ꝗᷓ", "qͦ",	"q̈", "ꝙ", "ꝙᷓ", ""],
  "r": ["r̄", "℞", "ꝛ", "ꝝ"],
  "s": ["ẜ"],
  "t": ["t"],
  "u": ["ū", "ꝰ"],
  "v": ["ꝟ", "v᷎", "v"],
  "w": ["wͭ"],
  "x": ["x̄"],
  "y": [""],
  "1": ["¹"],
  "2": ["²"],
  "3": ["³", "ʒ", "℥"],
  "4": ["⁴"],
  "5": ["⁵"],
  "6": ["⁶"],
  "7": ["⁷"],
  "8": ["⁸"],
  "9": ["⁹"],
  "0": ["⁰"],
  "-": ["̄", "̃", "ᷓ", "͛"],
  ",": ["̕", "̀"]
  },
	fontSize: $("body").css("fontSize").split("px",1).pop(),
	fontScale: {
		ltr: 0.8,
		rtl: 0.6
	},
  handleRadius: 0.3,
  colours: [
    ["polygon.empty", "fill", "Salmon"],
    ["polygon.empty", "opacity", "0.2"],
    ["polygon.ocr", "fill", "Blue"],
    ["polygon.ocr", "opacity", "0.2"],
    ["polygon.gt", "fill", "Green"],
    ["polygon.ocr", "opacity", "0.2"],
    ["polygon.comment", "fill-opacity", "0.2"],
    ["polygon.comment", "opacity", "0.6"],
    ["polygon.comment", "stroke", "Red"],
    ["polygon.comment", "stroke-width", "2px"],
    ["polygon.edit", "fill", "Orange"],
    ["polygon.edit", "fill-opacity", "0.4"],
    ["polygon.edit", "opacity", "0.4"],
  ]
};


function applyCSS(settings) {
  if ($("head style#CSS").length == 0){
    $('head').append('<style id="CSS" type="text/css"></style>');
  }
  $("head style#CSS").text(
  $.map(settings.colours, function(b){
    return `${b[0]} \{${b[1]}: ${b[2]}\}`
  }).join("\n"));
}


let Nashi = function() {};


Nashi.prototype.shortcuts = {
	global: function (e){
		let evt = e.originalEvent;
		let nsh = e.data;
		switch (evt.type){
			case "keyup":
			break;
			case "keydown":
				switch (evt.code) {

					case "Space":
						if (evt.ctrlKey && evt.shiftKey){
							evt.preventDefault();
							nsh.editor.zoom = !nsh.editor.zoom;
						} else if (evt.ctrlKey && nsh.editor.currentLine.length){
							evt.preventDefault();
							nsh.zoom();
						}
					break;

          case "Tab":
            if (evt.shiftKey){
              if (nsh.mode == "editLines" && nsh.editor.currentLine[0].previousSibling){
                nsh.editLine(nsh.editor.currentLine[0].previousSibling);
              }
            }else if(nsh.mode == "editLines" && nsh.editor.currentLine[0].nextSibling){
              nsh.editLine(nsh.editor.currentLine[0].nextSibling);
            }
            evt.preventDefault();

          break;

					case "KeyF":
						if (evt.ctrlKey){
							evt.preventDefault();
							$("#searchbar").toggleClass("active");
							if ($("#searchbar").hasClass("active")){
								$("#searchterm").focus();
							} else {
								nsh.editor.editor.focus();
							}
						}
					break;

					case "KeyS":
						if (evt.ctrlKey){
							e.preventDefault();
							nsh.downloadXML();
						}
					break;

          case "KeyE":
            if (evt.ctrlKey) {
              e.preventDefault();
              nsh.toggleEditMode();
            }
          break;

          case "Delete":
            if (nsh.mode == "editLines"){
              evt.preventDefault();
              if ($(".lastmoved", nsh.editor.svg0).length){
                nsh.delActivePoints();
              } else if ($(".edit", nsh.editor.svg0).length){
                nsh.delActiveLine();
              }
            }
          break;

					case "PageDown":
						if (evt.shiftKey){
							evt.preventDefault();
							nsh.changePage(1);
						}
					break;

					case "PageUp":
						if (evt.shiftKey){
							evt.preventDefault();
							nsh.changePage(-1);
						}
					break;
				}
			break;
		}
	},

	textinput: function (e){
		let evt = e.originalEvent;
		let nsh = e.data;
		switch (evt.type){
			case "keyup":
				if (nsh.editor.inputline.text()){
					let pad = parseFloat(nashi.editor.inputbox.css("paddingLeft").slice(0,-2))
									 +parseFloat(nashi.editor.inputbox.css("paddingRight").slice(0,-2));
					if ((nsh.editor.inputline.width() + pad) > nsh.editor.editor.width()){
						nsh.resizeFont(nsh.editor.editor.width() - pad);
					}
				}
				if (evt.code == "Insert" && nsh.editor.vkeyboard.css("display") != "none"){
					$("#vkeyboard").toggle(false);
				}
			break;
			case "keydown":
				switch (evt.code){
					case "Enter":
						if (document.activeElement == nsh.editor.inputline[0]){
							evt.preventDefault();
						}
						if (evt.shiftKey){
							evt.preventDefault();
							nsh.saveCurrentLine();
						}
					break;

					case "ArrowUp":
						if (evt.shiftKey){
							evt.preventDefault();
							nsh.editor.inputline.focus();
						}
					break;

					case "ArrowDown":
						if (evt.shiftKey){
							evt.preventDefault();
							nsh.editor.commentline.toggle(true);
							nsh.editor.commentline.focus();
						}
					break;

					case "Tab":
						if (document.activeElement == nsh.editor.commentline[0]){
							nsh.editor.inputline.focus();
						}
						if (evt.shiftKey){
							if (nsh.editor.currentLine[0].previousSibling){
		          	nsh.openLine(nsh.editor.currentLine[0].previousSibling);
							}
		        }else if(nsh.editor.currentLine[0].nextSibling){
		          nsh.openLine(nsh.editor.currentLine[0].nextSibling);
		        }
		        evt.preventDefault();
					break;

					case "Insert":
						if (evt.shiftKey){
							evt.preventDefault();
							let sel = window.getSelection();
							let pos = sel.focusOffset - 1;
							let text = sel.focusNode.data;
							if (pos < 0) pos = 0;
							let char = sel.focusNode.data[pos];
							if (char in nsh.settings.vKeyboard){
									let alt = nsh.settings.vKeyboard[char]
									let displaykeys = "";
									for (let i=0; i<alt.length; i++){ //"¹²³⁴⁵⁶⁷⁸⁹⁰"
											displaykeys += "<span class='key'>"+"①②③④⑤⑥⑦⑧⑨⓪"[i]
																			+ "</span><span>&nbsp;"+alt[i]+"</span> ";
									}
									nsh.editor.vkeyboard.html(displaykeys);
									nsh.editor.vkeyboard.toggle(true);
							}
						}
					break;
				}
				if (evt.code.startsWith("Digit") && nsh.editor.vkeyboard.is(":visible")){
					evt.preventDefault();
					let sel = window.getSelection();
					let pos = sel.focusOffset - 1;
					let text = sel.focusNode.data;
					if (pos < 0) pos = 0;
					let char = text[pos];
					if (char in nsh.settings.vKeyboard){
						let nmb = parseInt(evt.code[5]);
						if (nmb == 0){ nmb = 10; }
						if (nmb > nsh.settings.vKeyboard[char].length){ nmb = 1; }
						let insert = nsh.settings.vKeyboard[char][nmb-1];
						sel.focusNode.data = text.slice(0,pos)+insert+text.slice(pos+1);
						// reset caret
						let range = document.createRange();
						range.setStart(sel.focusNode, pos+insert.length);
						range.collapse(true);
						sel.removeAllRanges();
						sel.addRange(range);
					}
				}
			break;
		}
	}
};



Nashi.prototype.init = function(selector, settings=defaultSettings, page="_+first") {
	this.settings = settings;
  applyCSS(settings);
	selector.html(`
		<div id="editor">
				<svg id="svg0" xmlns="http://www.w3.org/2000/svg">
						<g id="group"
						 		onmouseup="$('#editor').trigger('svgevent', [evt])"
								onmousedown="$('#editor').trigger('svgevent', [evt])"
								onmousemove="$('#editor').trigger('svgevent', [evt])"
								onclick="$('#editor').trigger('svgevent', [evt])">
							<image id="image" height="0px" width="0px"/>
						</g>
				</svg>
				<svg id="svg1"><use href="#group"/></svg>
				<div id="inputbox">
						<span id="inputline" contenteditable="true"></span>
						<span id="commentline" contenteditable="true"></span>
						<span id="vkeyboard"></span>
				</div>
				<svg id="svg2"><use href="#group"/></svg>
		</div>
		`);
	this.editor = {
		editor: $("#editor", selector),
		inputbox: $("#inputbox", selector),
		inputline: $("#inputline", selector),
		commentline: $("#commentline", selector),
		vkeyboard: $("#vkeyboard", selector),
		image: $("#image", selector),
		svg0: $("#svg0", selector),
		svg1: $("#svg1", selector),
		svg2: $("#svg2", selector)
	};

  [this.editor.svg1, this.editor.svg2].forEach(function(svg){
		svg.setViewbox = function(x, y, width, height){
			this[0].setAttribute("viewBox", [x, y, width, height].join(","));
			return this;
		};
		svg.getViewbox = function(){
			return this[0].getAttribute("viewBox").split(",").map(x => parseInt(x));
		};
	});

	// Events don't bubble through the "use", so we're redirecting custom events.
	this.editor.editor.on("svgevent", this, this.svgEvent);

	this.editor.editor.on("pageturn", this.onPageTurn);
	this.editor.editor.on("linechange", this.onLineChange)

	this.getData(page);
	this.mode = "transcribe";
  this.edits = [];
	this.editor.scale = 1;
	this.editor.zoom = false;
  this.editor.drawLine = null;
  this.editor.selection = null;

	$( document ).on("keydown keyup", this, this.shortcuts.global);
	this.editor.inputbox.on("keydown keyup", this, this.shortcuts.textinput);

	this.editor.commentline.on("blur", this, this.commentlineOnBlur);
};


Nashi.prototype.onPageTurn = function(evt,  pagenum){
	$(".pagenum").text(pagenum);
	$(".linenum").text();
	$("#sidebar li").each(function(n, li){
		$(li).toggleClass("active", (li.innerText.split(" ")[0] == pagenum));
	});
};


Nashi.prototype.onLineChange = function(evt, linenum){
	$(".linenum").text(linenum);
};


Nashi.prototype.downloadXML = function () {
    window.open(this.pagedata["page"] + '/data?download=xml');
};


Nashi.prototype.downloadIMAGE = function () {
	window.open(this.pagedata.image.file);
};


Nashi.prototype.changePage = function(dir){
	this.editor.inputbox.toggle(false);
	switch (dir) {
		case -1:
			this.getData(this.pagedata.page + "+prev");
			break;
		default:
			this.getData(this.pagedata.page + "+next");
	}
};


Nashi.prototype.saveCurrentLine = function(){
	this.editor.currentLine.toggleClass("saving", true);
	this.editor.currentLine.toggleClass("gt", true);
	this.pagedata.lines[this.editor.currentLine[0].id].text.content = this.editor.inputline.text();
	this.pagedata.lines[this.editor.currentLine[0].id].text.status = "edit";
	if (this.editor.commentline.css("display") != "none"
			&& document.activeElement == this.editor.commentline[0]){
			this.saveCommentLine();
			// prevent subsequent blur event from saving next line
			this.editor.commentline.data("dontsave", true);
	}
	this.sendLine(this.editor.currentLine[0]);
	if (this.editor.currentLine[0].nextElementSibling &&
			this.editor.currentLine[0].nextElementSibling.tagName == "polygon"){
				this.openLine(this.editor.currentLine[0].nextElementSibling);
	}
};


Nashi.prototype.saveCommentLineOld = function(){
	let cur = this.editor.currentLine[0].id;
  if (this.editor.commentline.data("dontsave")){
      // saving already done in savetext()
      this.editor.commentline.data("dontsave", false);
  } else {
    if (this.pagedata.lines[cur].comments != $("#commentline").html()){
      this.pagedata.lines[cur].comments = $("#commentline").html();
	    var tgl = this.editor.commentline.html() ? true : false;
      this.editor.currentLine.toggleClass("comment", tgl);
      this.editor.currentLine.toggleClass("saving", true);
      this.sendLine(this.editor.currentLine[0]);
    }
  }
};

Nashi.prototype.saveCommentLine = function(){
	let cur = this.editor.currentLine[0].id;
  if (this.editor.commentline.data("dontsave")){
      // saving already done in savetext()
      this.editor.commentline.data("dontsave", false);
  } else {
    let content = $("<div>"+this.editor.commentline.html().replace("<br>", "\n")+"</div>")
                  .text().replace("\n", "<br>")
    if (this.pagedata.lines[cur].comments != content){
      this.pagedata.lines[cur].comments = content;
	    var tgl = content ? true : false;
      this.editor.currentLine.toggleClass("comment", tgl);
      this.editor.currentLine.toggleClass("saving", true);
      this.sendLine(this.editor.currentLine[0]);
    }
  }
};


Nashi.prototype.commentlineOnBlur = function(evt){
	let nsh = evt.data;
	nsh.saveCommentLine();
};


Nashi.prototype.sendLine = function(polygon){
	let data = {edits: []}
			page = this.pagedata.page;
	data["edits"].push({
			id: polygon.id,
			action: "change",
			input:  this.pagedata.lines[polygon.id] });
	data = JSON.stringify(data);
	$.ajax({
		url: page + '/data',
		type: 'POST',
		contentType: 'application/json;charset=UTF-8',
		data: data,
		success: function( data ) {
			$("#sidebar a").each(function(_, a){
				if(a.text.startsWith(page)){
					a.lastChild.innerText = data.lineinfo;
					let alltranscr = (function(a,b){return a==b})(...data.lineinfo.split("/"))
					let anytranscr = (function(a,b){return a<b && a > 0})(...data.lineinfo.split("/"));
					$(a.lastChild).toggleClass("label-success", alltranscr);
					$(a.lastChild).toggleClass("label-info", anytranscr);
					$(a.lastChild).toggleClass("label-default", !anytranscr && !alltranscr);
				}
			});
			$(polygon).toggleClass("saving", false);
		}
	});
};

Nashi.prototype.getData = function(filename, line=""){
	if (this.mode == "editLines") this.toggleEditMode();
  this.editor.inputbox.toggle(false);
  $.ajax({
		url: filename + "/data",
		contentType: "application/json;charset=UTF-8",
		context: this,
		success: function( data ) {
			$("polygon", nashi.editor.svg0).remove();
			this.pagedata = data;

			this.editor.editor.trigger("pageturn", [data.page]);

			let x = data.image.image_x;
			let y = data.image.image_y;
			this.editor.scale = this.editor.editor.outerWidth() / x;

			this.editor.editor.height(this.editor.scale * y);

			this.editor.editor.css("direction", data.direction);
			this.editor.inputbox.css("direction", data.direction);

			this.editor.image.attr({
				x: 0,
				y: 0,
				width: x,
				height: y,
				href: data.image.file
			});

			this.editor.svg1.attr({
				width: this.editor.scale * x,
				height: this.editor.scale * y,
				viewBox: "0,0,+"+x+"," + y,
				preserveAspectRatio: "xMinYMax slice"
			});

			this.editor.svg2.attr({
				width: this.editor.scale * x,
				height: "0",
				viewBox: "0,0,0,0",
				preserveAspectRatio: "xMinYMin slice"}
			);

			let svgns = this.editor.svg0.attr("xmlns");
			let group = $("#group", this.editor.svg0);
      $.each(data.regions, function(id, d){
          group.append($(document.createElementNS(svgns, "polygon")).attr({
            id: id,
            points: d.points,
            class: "region"
          })
        );
      });
	    $.each(data.lines, function(id, d){
					group.append($(document.createElementNS(svgns, "polygon")).attr({
						id: id,
						points: d.points,
						class: "line " + d.text.status
					}).toggleClass("comment", (d.comments != ""))
				);
	    });
      if (line) {
        this.openLine($("#"+line, this.editor.svg0)[0])
        window.scrollTo(0, this.editor.inputbox.offset().top - 150);
      }
  	}
	});
};


Nashi.prototype.svgEvent = function(e, event) {
	// Nashi/this is here t.data, this the primary target!
	let nsh = e.data;
	switch (nsh.mode){
		case "transcribe":
			switch (event.type) {
				case "click":
					switch (event.target.tagName){
						case "polygon":
              if (event.target.classList.contains("line")){
							  nsh.openLine(event.target);
							  break;
              }
					}
				break;
			}
		break;

		case "editLines":
			switch (event.type) {
				case "click":
					switch (event.target.tagName){
						case "polygon":
							nsh.editLine(event.target);
							break;
					}
				break;
				case "mousedown":
          let cl = event.target.classList;
          if (event.shiftKey){
            nsh.startSelecting(event);
					} else if (cl.contains("pointhandle")){
						nsh.activatePointHandle(event.target);
					} else if (cl.contains("linehandle")){
            nsh.createPoint(event);
          } else if (cl.contains("region") && cl.contains("edit")) {
            nsh.createLine(event);
          }
				break;
				case "mousemove":
					if (nsh.editor.activePoint){
						nsh.moveActivePoint(event);
					} else if (nsh.editor.drawLine){
            nsh.drawPolygon(event);
          } else if (nsh.editor.selection){
            nsh.drawSelection(event);
          }
				break;
				case "mouseup":
					if (nsh.editor.activePoint){
						nsh.dropPoint();
					} else if (nsh.editor.drawLine){
            nsh.finishPolygon(event);
          } else if (nsh.editor.selection){
            nsh.endSelection(event);
          }
				break;
			}
		break;

		case "editRegions":
		break;
	}
};


Nashi.prototype.openLine = function(polygon) {
	this.editor.editor.trigger("linechange", [polygon.id]);
	this.editor.currentLine = $("#"+polygon.id, this.editor.svg0);

	let xpoints = $(polygon.points).toArray().map(function(x){return x.x});
	let ypoints = $(polygon.points).toArray().map(function(x){return x.y});
	let text = this.pagedata.lines[polygon.id].text.content;
	let coms = this.pagedata.lines[polygon.id].comments;

	this.editor.inputline.text(text);
	this.editor.commentline.html(coms);
	this.editor.vkeyboard.toggle(false);

	this.editor.commentline.toggle(coms != "");

	this.toggleDivision(Math.min(...xpoints), Math.max(...xpoints),
											Math.min(...ypoints), Math.max(...ypoints));

	if (this.editor.zoom){this.zoom();}

	this.editor.inputline.focus();
};


Nashi.prototype.toggleDivision = function(xmin, xmax, ymin, ymax) {
	this.editor.inputbox.toggle();
	if (this.editor.inputbox.css("direction") == "rtl"){
		let padright = this.editor.scale * (this.editor.image.attr("width") - xmax);
		this.editor.inputbox.css("paddingRight", padright + "px");
		this.editor.inputbox.css("paddingLeft", "2px");
	} else {
		let padleft = this.editor.scale * xmin;
		this.editor.inputbox.css("paddingRight", "2px");
		this.editor.inputbox.css("paddingLeft", padleft + "px");
	}
	if (this.editor.svg1.getViewbox()[3] != ymax ||
			(this.editor.zoom != true && this.editor.svg1.getViewbox()[0] != 0)){
		this.splitSVG(ymax);
		let fs = this.settings.fontScale[this.editor.inputbox.css("direction")];
		this.editor.inputbox.css("fontsize",
														 (ymax-ymin) * fs * this.editor.scale + "px");
		this.editor.inputbox.toggle(true);
		if (this.editor.inputline.text()){
			this.resizeFont(this.editor.scale * (xmax-xmin));
		} else {
			this.editor.inputbox.css("fontSize", this.settings.fontSize + "px");
		}
	}
};


Nashi.prototype.splitSVG = function(y) {
	let img_x = this.editor.image.attr("width");
	let img_y = this.editor.image.attr("height");
	this.editor.svg1.attr("height", Math.round(this.editor.scale * y));
	this.editor.svg1.setViewbox(0, 0, img_x, y);
	this.editor.svg2.attr("height", Math.round(this.editor.scale * (img_y - y)));
	this.editor.svg2.setViewbox(0, y, img_x, img_y);
};


Nashi.prototype.resizeFont = function(len) {
	let fs = parseFloat(this.editor.inputbox.css("fontSize").split("px", 1).pop());
	let fac = len / this.editor.inputline.outerWidth();
	this.editor.inputbox.css("fontSize", fac * fs + "px");
};


Nashi.prototype.zoom = function() {
	let xpoints = $(this.editor.currentLine[0].points).toArray().map(function(x){return x.x});
	let ypoints = $(this.editor.currentLine[0].points).toArray().map(function(x){return x.y});
	let xmin = Math.min(...xpoints),
   		xmax = Math.max(...xpoints),
   		ymin = Math.min(...ypoints),
   		ymax = Math.max(...ypoints);
	this.editor.svg1.setViewbox(xmin, 0, xmax-xmin, ymax);
	if ((ymax-ymin) / (xmax-xmin) > this.editor.svg1.height() / this.editor.svg1.width()) {
		this.editor.svg1.attr( "height", ((ymax-ymin) / (xmax-xmin)) * this.editor.svg1.width());
	}
	this.editor.svg2.setViewbox(xmin, ymax, xmax-xmin, this.editor.image.attr("height"));

	var newscale = this.editor.editor.outerWidth() / (xmax-xmin);
	this.editor.inputbox.css({paddingLeft:  "2px", paddingRight:  "2px"});
	let fs = this.settings.fontScale[this.editor.inputbox.css("direction")];
	this.editor.inputbox.css("font-size", (ymax-ymin)* fs * newscale + "px");
	if (this.editor.inputline.text()) {
		this.resizeFont(newscale * (xmax-xmin));
	}
};


///////////// EDIT MODE ////////////////////////////////////////////////////////


Nashi.prototype.toggleEditMode = function(){
  if (this.mode != "editLines"){
    this.mode = "editLines";
    this.edits = [];
  } else {
    this.mode = "transcribe";
    $("#pointhandles", this.editor.svg0).remove();
    $(".edit", this.editor.svg0).toggleClass("edit", false);
    if ($.isEmptyObject(this.edits)){
      // no changes made
    } else {
      if (confirm("Save changes")){
        this.saveEdits();
      } else {
        this.getData(this.pagedata.page); //reload current page
      }
    }
  }
};


Nashi.prototype.sanitizePoints = function(pointstring){
	// for some reason, chrome writes points without "," to the points attribute
	if (pointstring.includes(",")){
		return pointstring;
	} else {
	return $.trim(pointstring.split(" ").map((x, n) => {switch(n%2){
		case 1: return ","+x; case 0: return " "+x}}).join(""))
	}
}


Nashi.prototype.saveEdits = function(){
  let pagename = this.pagedata.page,
      edited = new Set($.map(this.edits, e => e["id"]))
			edits = [];

  $("polygon", this.editor.svg0)
    .filter(function(n,e){return edited.has(e.id)})
		.toggleClass("saving", true);

  edited.forEach(function(id){
    this.edits.forEach(function(edit){
      if (edit.id == id){
				let action = edit["action"];
				if (action == "delete"){
					let shortlived = false;
					for (let i = edits.length - 1; i >= 0; i--){
						if (edits[i]["id"] == id && edits[i]["action"] == "create"){
							edits = edits.slice(0,i);
							shortlived = true;
						}
					}
					if (!shortlived){
						edits.push({
							action: action,
							id: id,
							input: edit.line
						})
					};
				} else {
					let pos = -1;
					$.each(edits, function(e){if (edits[e]["id"] == id){pos = e;}});
					if (pos < 0 || edits[pos]["action"] != "change" || action == "create"){
						pos = edits.length;
					}
					edits[pos] = {
						action: action,
						id: id,
						input: edit.line
					}
				}
      }
    }, this);
	}, this);
	
  let data = JSON.stringify({edits: edits});
  $.ajax({
    url: pagename + '/data',
    type: 'POST',
    contentType: 'application/json;charset=UTF-8',
    data: data,
    success: function( data ) {
      $("#sidebar a").each(function(_, a){
        if(a.text.startsWith(pagename)){
          a.lastChild.innerText = data.lineinfo;
          var alltranscr = (function(a,b){return a==b})(...data.lineinfo.split("/"))
          var anytranscr = (function(a,b){return a<b && a > 0})(...data.lineinfo.split("/"));
          $(a.lastChild).toggleClass("label-success", alltranscr);
          $(a.lastChild).toggleClass("label-info", anytranscr);
          $(a.lastChild).toggleClass("label-default", !anytranscr && !alltranscr);
        }
      });
      $(".saving").toggleClass("saving", false);
    }
  });
};


Nashi.prototype.pushEdit = function(type, lid, rid){
  switch (type) {
    case "delete":
      this.edits.push({
    		"id": lid,
    		"action": "delete",
    		"line": $.extend(true, {}, this.pagedata.lines[lid])
    	});
      delete this.pagedata.lines[lid];
    break;

    case "create":
      this.pagedata.lines[lid] = {
    		comments: "",
    		points: this.sanitizePoints($("#"+lid, this.editor.svg0).attr("points")),
    		region: rid,
    		text: {content: "", status: "empty"}
    	};
      this.edits.push({
    		"id": lid,
    		"action": "create",
    		"line": $.extend(true, {}, this.pagedata.lines[lid])
    	});
    break;

    case "change":
      this.pagedata.lines[lid]["points"] = this.sanitizePoints($("#"+lid, this.editor.svg0).attr("points"));
      this.edits.push({
        "id": lid,
        "action": "change",
        "line": $.extend(true, {}, this.pagedata.lines[lid])
      });
    break;
  }
};


Nashi.prototype.drawLineHandles = function(polygon, target){
	$(".linehandle", this.editor.svg0).remove();
	let svgns = this.editor.svg0.attr("xmlns");
	let radius = this.settings.handleRadius / (this.editor.editor.width() / this.editor.image.attr("width")) + "em";
	for (let n = 0; n < polygon.points.length; n++){
		let newelement = document.createElementNS(svgns, 'line');
		$(newelement).attr({
			class: "linehandle",
			x1: polygon.points[n].x,
			y1: polygon.points[n].y,
			x2: polygon.points[(n+1) % polygon.points.length].x,
			y2: polygon.points[(n+1) % polygon.points.length].y,
			"stroke-width": radius,
			stroke: "transparent",
		});
		target.prepend(newelement);
	}
};


Nashi.prototype.editLine = function(polygon){
	this.editor.editor.trigger("linechange", [polygon.id]);
	this.editor.currentLine = $("#"+polygon.id, this.editor.svg0);

	$(".edit", this.editor.editor).toggleClass("edit", false);
	this.editor.currentLine.toggleClass("edit", true);

  $("#pointhandles", this.editor.svg0).remove();

	let svgns = this.editor.svg0.attr("xmlns");
	let group = $("#group", this.editor.svg0);
	let newgroup = document.createElementNS(svgns, 'g');
	newgroup.setAttribute("id","pointhandles");
	group.append(newgroup);

	this.drawLineHandles(polygon, $(newgroup));

	let radius = this.settings.handleRadius / (this.editor.editor.width() / this.editor.image.attr("width")) + "em";
	$(polygon.points).each(function(n, p){
		let newelement = document.createElementNS(svgns, 'circle');
		$(newelement).attr({
			id: polygon.id + "_p" + n,
			class: "pointhandle",
			cx: p.x,
			cy: p.y,
			r: radius
		});
		$(newgroup).append(newelement);
	});
};


Nashi.prototype.activatePointHandle = function(circle){
	for (p of this.editor.currentLine[0].points){
		if (p.x == circle.cx.baseVal.value && p.y == circle.cy.baseVal.value){
			this.editor.activePoint = p;
		}
	}
  handle = $("#"+circle.id, this.editor.svg0);
	handle.toggleClass("movehandle", true);
	$(".lastmoved", this.editor.editor).toggleClass("lastmoved", false);
	handle.toggleClass("lastmoved", true);
	this.editor.editor.css("cursor", "move");
	this.editor.activeHandle = handle;
};


Nashi.prototype.dropPoint = function(){
	this.editor.editor.css("cursor", "auto");
	this.editor.activePoint = null;
	$(".movehandle", this.editor.editor).toggleClass("movehandle", false);
	this.drawLineHandles(this.editor.currentLine[0], $("#pointhandles", this.editor.svg0));

  this.pushEdit("change", this.editor.currentLine[0].id, "");
};


// calculate point on image from event
Nashi.prototype.getImgPoint = function(evt){
	let p = this.editor.svg0[0].createSVGPoint();
	p.x = evt.clientX
	p.y = evt.clientY
	let m = evt.target.getScreenCTM();
	p = p.matrixTransform(m.inverse());
	p.x = Math.round(p.x);
	p.y = Math.round(p.y);
	return p;
};


Nashi.prototype.moveActivePoint = function(evt){
	let c = this.getImgPoint(evt);
	this.editor.activeHandle.attr({cx: c.x, cy: c.y});
	this.editor.activePoint.x = c.x;
	this.editor.activePoint.y = c.y;
};


Nashi.prototype.drawPolygon = function(evt){
  let pos = this.getImgPoint(evt);
  let pp = this.editor.drawLine[0].points;
  pp[1].x = pp[2].x = pos.x;
	pp[3].y = pp[2].y = pos.y;
};


Nashi.prototype.finishPolygon = function(evt){
  this.editor.drawLine = null;
	let lid = this.editor.currentLine[0].id;
	if ($.unique($.map(nashi.editor.currentLine[0].points, x => x.x)).length < 2 
		  || $.unique($.map(nashi.editor.currentLine[0].points, x => x.y)).length < 2){
				// Line without extension, delete
				$("#pointhandles", this.editor.svg0).remove();
				this.editor.currentLine.remove();
				this.editor.currentLine = null;
		} else {
			this.pushEdit("create", lid, lid.split("_")[0]);
			this.editLine(this.editor.currentLine[0]);
		}
};


Nashi.prototype.createLine = function(evt){
  let startp = this.getImgPoint(evt);
  $(".pointhandle", this.editor.svg0).remove();
  $(".linehandle", this.editor.svg0).remove();
  $("polygon.edit", this.editor.svg0).toggleClass("edit", false);

  let existing = $(".line", this.editor.svg0)
    .filter(function(n, e){return e.id.startsWith(evt.target.id)})
    .map(function(n, e){return e.id.split("_l")[1]})
    .toArray();
  let len = Math.min(...existing.map(function(n){return n.length}));
  let lineno = Math.max(...existing) + 1;
  let svgns = this.editor.svg0.attr("xmlns");
  if (len == Infinity){len = 3};
	if (lineno == -Infinity){lineno = 0};
  let pid = evt.target.id + "_l" + lineno.toString().padStart(len, "0");

  $("#group", this.editor.svg0).append(
    $(document.createElementNS(svgns, "polygon")).attr({
      id: pid,
      points: ([startp.x,startp.y].join()+" ").repeat(4).slice(0,-1),
      class: "line empty edit"
    })
  );
  let cur = $("#"+pid, this.editor.svg0);
  this.editor.drawLine = cur;
  this.editor.currentLine = cur;
};


Nashi.prototype.createPoint = function(evt){
  let c = this.getImgPoint(evt);
  $(".lastmoved", this.editor.svg0).toggleClass("lastmoved", false);
  let polygon = this.editor.currentLine[0];
  let svgns = this.editor.svg0.attr("xmlns");
  let newelement = document.createElementNS(svgns, 'circle');
  let radius = this.settings.handleRadius / (this.editor.editor.width() / this.editor.image.attr("width")) + "em";
  $(newelement).attr({
    id: polygon.id + "_p" + polygon.points.length,
    class: "pointhandle movehandle lastmoved",
    cx: c.x,
    cy: c.y,
    r: radius
  });
  $("#pointhandles", this.editor.svg0).append(newelement);

  for(let n=0; n<this.editor.currentLine[0].points.length; n++){
    let p = this.editor.currentLine[0].points.getItem(n);
    if (p.x == evt.target.x2.baseVal.value && p.y == evt.target.y2.baseVal.value){
      this.editor.currentLine[0].points.insertItemBefore(c, n);
      this.editor.activePoint = c;
      break;
    }
  }

  this.editor.editor.css("cursor", "move");
	this.editor.activeHandle = $("#"+newelement.id, this.editor.svg0);
};


Nashi.prototype.endSelection = function(evt){
  this.editor.selection.remove();
  this.editor.selection = null;
};


Nashi.prototype.drawSelection = function(evt){
  let pp = this.editor.selection[0].points;
  let pos = this.getImgPoint(evt);
  pp[1].x = pp[2].x = pos.x;
  pp[3].y = pp[2].y = pos.y;
  let xr = [pp[0].x, pos.x].sort(function(a, b){return a-b});
  let yr = [pp[0].y, pos.y].sort(function(a, b){return a-b});
  $(".pointhandle", this.editor.svg0).each(function(n, phand){
    let px = phand.cx.baseVal.value;
    let py = phand.cy.baseVal.value;
    $(phand).toggleClass("lastmoved",
      (xr[0] < px && px < xr[1] && yr[0] < py && py < yr[1])
      );
  });
};


Nashi.prototype.startSelecting = function(evt){
  let startp = this.getImgPoint(evt);
  let svgns = this.editor.svg0.attr("xmlns")
  $("#pointhandles", this.editor.svg0).append(
    $(document.createElementNS(svgns, 'polygon')).attr({
      id: 'select',
      points: ([startp.x,startp.y].join()+" ").repeat(4).slice(0,-1),
      class: 'select'
    })
  );
  this.editor.selection = $("#select", this.editor.svg0);
};


Nashi.prototype.delActiveLine = function(){
  $("#pointhandles", this.editor.svg0).remove();
  this.pushEdit("delete", this.editor.currentLine[0].id, "");
  this.editor.currentLine.remove();
  this.editor.currentLine = null;
};


Nashi.prototype.delActivePoints = function(){
  $(".lastmoved", this.editor.svg0).toArray().forEach(function(h){
      let plist = this.editor.currentLine[0].points;
      for (var j=plist.length; j--;){
        if (h.cx.baseVal.value === plist[j].x && h.cy.baseVal.value === plist[j].y){
          plist.removeItem(j);
          break;
        }
      }
    h.remove();
  }, this);
  if (this.editor.currentLine[0].points.length < 3){
    this.delActiveLine();
  } else {
    this.pushEdit("change", this.editor.currentLine[0].id, "");
    this.editLine(this.editor.currentLine[0]);
  }
};


/////SEARCH FUNCTIONS///////////////////////////////////////////////////////////

Nashi.prototype.jump_comment = function(start="", dir=1){
  let lines = Object.keys(this.pagedata.lines);
  let found = false;
  console.log(start)
  $("#searchmessage").text("Searching…");
  if (!start){start=lines[0];}
  if (dir==-1){lines.reverse();}
  console.log("start search at " + start);
  for (var ix=lines.indexOf(start)+1; ix<lines.length; ix++){
    if (this.pagedata.lines[lines[ix]].comments != ""){
      this.openLine($("#"+lines[ix], this.editor.svg0)[0]);
      this.editor.inputbox.toggle(true);
      window.scrollTo(0, this.editor.inputbox.offset().top - 150);
      found = true;
      $("#searchmessage").text("");
      break;
    }
  }
  if (!found){
		$("#searchmessage").text("Searching on other pages...");
		data = {dir: dir}
    data = JSON.stringify(data);
    $.ajax({
			url: this.pagedata.page + '/comments_jump',
			type: 'POST',
			contentType: 'application/json;charset=UTF-8',
      context: this,
			data: data,
			success: function( data ) {
				if (data.result.page){
					found = true;
					$("#searchmessage").text("");
          this.editor.inputbox.toggle(false);
          $("#group polygon", this.editor.svg0).remove();
          this.getData(data.result.page, line=data.result.line);
				} else {	$("#searchmessage").text("Not found."); }
	     }
		});
	}
};


Nashi.prototype.search = function(st="", start="", dir=1, comments=false){
  let lines = Object.keys(this.pagedata.lines);
  let found = false;
  console.log(start)
  $("#searchmessage").text("Searching…");
  if (!start){start=lines[0];}
  if (dir==-1){lines.reverse();}
  console.log("start search at " + start);
	for (var ix=lines.indexOf(start)+1; ix<lines.length; ix++){
		if (this.pagedata.lines[lines[ix]].text.content.includes(st)){
			this.openLine($("#"+lines[ix], this.editor.svg0)[0]);
      this.editor.inputbox.toggle(true);
      window.scrollTo(0, this.editor.inputbox.offset().top - 150);
			found = true;
			$("#searchmessage").text("");
			break;
		}
		if (comments && this.pagedata.lines[lines[ix]].comments.includes(st)){
      this.openLine($("#"+lines[ix], this.editor.svg0)[0]);
      this.editor.inputbox.toggle(true);
      window.scrollTo(0, this.editor.inputbox.offset().top - 150);
			found = true;
			$("#searchmessage").text("");
			break;
		}
	}
	if (!found){
		$("#searchmessage").text("Searching on other pages...");
		data = {searchterm: st, dir: dir, comments: comments}
    data = JSON.stringify(data);
    $.ajax({
			url: this.pagedata.page + '/search_continue',
			type: 'POST',
			contentType: 'application/json;charset=UTF-8',
      context: this,
			data: data,
			success: function( data ) {
				if (data.result.page){
					found = true;
					$("#searchmessage").text("");
          this.editor.inputbox.toggle(false);
          $("#group polygon", this.editor.svg0).remove();
          this.getData(data.result.page, line=data.result.line);
				} else {	$("#searchmessage").text("Not found."); }
	     }
		});
	}
};


Nashi.prototype.drawROrder = function(){
  let rids = Object.keys(this.pagedata.regions).sort();
  let rcorners =  rids.map(rid => {
    return $($("#"+rid, this.editor.svg0)[0].points).toArray()
      .sort(function(a,b){return (a.x+a.y)-(b.x+b.y)})[0]
  });
  let points = rcorners.map(p => {return `${p.x},${p.y}`}).join(" ");
  let svgns = this.editor.svg0.attr("xmlns");
  let radius = this.settings.handleRadius / (this.editor.editor.width() / this.editor.image.attr("width")) + "em";
  let marker = document.createElementNS(svgns, 'marker');
  let mpath = document.createElementNS(svgns, 'path');
  let defs = document.createElementNS(svgns, 'defs')
  $(marker).attr({
    id: "ar",
    viewBox: "0 0 15 15",
    refX: "5",
    refY: "5",
    markerUnits: "strokeWidth",
    markerWidth: "6",
    markerHeight: "6",
    orient: "auto-start-reverse"
  });
  $(mpath).attr({d: "M 0 0 L 10 5 L 0 10 z", fill: "black"});

  $(marker).append(mpath);
  $(defs).append(marker);
  this.editor.svg2.prepend(defs); // Has to be in visible svg...
  this.editor.svg1.prepend(defs);

  let newelement = document.createElementNS(svgns, 'polyline');
  $(newelement).attr({
    class: "rOrder",
    points: points,
    fill: "none",
    stroke: "black",
    "stroke-width": radius,
    "marker-mid": "url(#ar)",
    "marker-end": "url(#ar)"
  });
  $("#group", this.editor.svg0).append(newelement);
}
