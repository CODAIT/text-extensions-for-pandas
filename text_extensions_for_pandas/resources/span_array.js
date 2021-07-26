// Increment the version to invalidate the cached script
const VERSION = 0.79
const global_stylesheet = document.head.querySelector("style.span-array-css")
const local_stylesheet = document.currentScript.parentElement.querySelector("style.span-array-css")

if(window.SpanArray == undefined || window.SpanArray.VERSION == undefined || window.SpanArray.VERSION < VERSION) {

    // Replace global SpanArray CSS with latest copy
    if(local_stylesheet != undefined) {
        if(global_stylesheet != undefined) {
            document.head.removeChild(global_stylesheet)
        }
        document.head.appendChild(local_stylesheet)
    }

    // Sets up the SpanArray global namespace
    window.SpanArray = {}
    window.SpanArray.VERSION = VERSION

    window.SpanArray.TYPE_OVERLAP = 0;
    window.SpanArray.TYPE_NESTED = 1;
    window.SpanArray.TYPE_COMPLEX = 2;
    window.SpanArray.TYPE_SOLO = 3;

    const TYPE_OVERLAP = window.SpanArray.TYPE_OVERLAP;
    const TYPE_NESTED = window.SpanArray.TYPE_NESTED;
    const TYPE_COMPLEX = window.SpanArray.TYPE_COMPLEX;
    const TYPE_SOLO = window.SpanArray.TYPE_SOLO;

    function sanitize(input) {
        let out = input.slice();
        out = out.replace(/&/g, "&amp;")
        out = out.replace(/</g, "&lt;")
        out = out.replace(/>/g, "&gt;")
        out = out.replace(/\$/g, "<span>&#36;</span>")
        out = out.replace(/"/g, "&quot;")
        out = out.replace(/(\r|\n)/g, "<br>")
        return out;
    }

    /** Comparison function used to sort SpanArrays by position and length
     *  Will sort by primarily by earliest beginning point. On a tie, will prioritize latest end point (first and largest)
     *  Used by mark relationship algorithm.
     */
    function compareSpanArrays(a, b) {
        const start_diff = a.begin - b.begin
        if(start_diff == 0) {
            return b.end - a.end
        }
        return start_diff;
    }

    /** Models an instance of a SpanArray, with document-separated spans and text
     * NOTE: Using docs instead of documents to avoid unintentionally manipulating the global 'document' object.
    */
    class SpanArray {
        constructor(docs, show_offsets, script_context) {
            this.docs = docs
            this.show_offsets = show_offsets
            this.script_context = script_context

            // For each doc, generate a lookup map for quick ID access
            this.docs = this.docs.map(doc => {
                doc.span_objects = Span.arrayFromSpanArray(doc.doc_spans)
                doc.lookup_table = {}
                doc.span_objects.forEach(span => {
                    doc.lookup_table[span.id] = span
                })
                return doc
            })
        }

        render() {
            let span_array_frag = document.createDocumentFragment()
            // For each document, create a document fragment and append to a document container
            for(let doc_index = 0; doc_index < this.docs.length; doc_index++)
            {
                let doc = this.docs[doc_index]
                let doc_container = document.createElement("div")
                // Using the data-doc-id attribute allows a selector to access a document's render by its index
                doc_container.setAttribute("data-doc-id", doc_index)
                doc_container.classList.add("document")
                const document_fragment = getDocumentFragment(doc, this.show_offsets)
                if(this.show_offsets) {
                    attachDocumentEvents(document_fragment, doc, this)
                }
                doc_container.appendChild(document_fragment)
                span_array_frag.appendChild(doc_container)
            }
            let container = this.script_context.parentElement.querySelector(".span-array")
            if(container != undefined) {
                container.innerHTML = ""
                container.appendChild(span_array_frag)
            } else {
                console.error("No container found for SpanArray renderer")
            }
        }
    }

    window.SpanArray.SpanArray = SpanArray


    /** Models an instance of a Span and its relationship to other spans in the document */
    class Span {

        // Creates an ordered list of entries from a list of spans with struct [begin, end]
        static arrayFromSpanArray(spanArray) {
            let entries = []
            let span;
            for(let i = 0; i < spanArray.length; i++)
            {
                span = spanArray[i];
                entries.push(new Span(i, span[0], span[1]))
            }

            entries = entries.sort(compareSpanArrays)

            let set;
            for(let i = 0; i < entries.length; i++) {
                for(let j = i+1; j < entries.length && entries[j].begin < entries[i].end; j++) {
                    if(entries[j].end <= entries[i].end) {
                        set = {type: TYPE_NESTED, entry: entries[j]}
                    } else {
                        set = {type: TYPE_OVERLAP, entry: entries[j]}
                    }
                    entries[i].sets.push(set)
                }
            }

            return entries
        }

        constructor(id, begin, end) {
            this.id = id
            this.begin = begin
            this.end = end
            this.sets = []
            this.visible = true
            this.highlighted = false
        }

        // Returns only visible sets
        get valid_sets() {
            let valid_sets = []

            this.sets.forEach(set => {
                if(set.entry.visible) valid_sets.push(set)
            })

            return valid_sets
        }

        // Returns true if mark should render as a compound set of spans
        isComplex() {
            for(let i = 0; i < this.valid_sets.length; i++) {
                let otherMember = this.valid_sets[i].entry;
                if(this.valid_sets[i].type == TYPE_OVERLAP && otherMember.visible) {
                    return true;
                } else {
                    if(otherMember.valid_sets.length > 0 && otherMember.visible) {
                        return true;
                    }
                }
            }
            return false;
        }

        // Gets the combined span of all connected elements
        getSetSpan() {
            let begin = this.begin
            let end = this.end
            let highest_id = this.id

            this.valid_sets.forEach(set => {
                let other = set.entry.getSetSpan()
                if(other.begin < begin) begin = other.begin
                if(other.end > end) end = other.end
                if(other.highest_id > highest_id) highest_id = other.highest_id
            })

            return {begin: begin, end: end, highest_id: highest_id}
        }
    }

    window.SpanArray.Span = Span

    // Get the DocumentFragment for a single document 
    function getDocumentFragment(doc, show_offsets) {

        const doc_text = doc.doc_text;
        const entries = doc.span_objects;

        let frag = document.createDocumentFragment()
        
        // Render Table
        if(show_offsets) {
            let table = document.createElement("table")
            table.innerHTML = `
            <thead>
            <tr>
                <th></th>
                <th></th>
                <th>begin</th>
                <th>end</th>
                ${(doc['doc_token_spans'] != undefined) ? '<th>begin token</th> <th>end token</th>' : ''}
                <th>context</th>
            </tr>
            </thead>`
            let tbody = document.createElement("tbody")
            entries.forEach(entry => {
                let row = document.createElement("tr")
                row.setAttribute("data-id", entry.id.toString())
                if(!entry.visible)
                {
                    row.classList.add("disabled")
                }
                if(entry.highlighted)
                {
                    row.classList.add("highlighted")
                }

                // Adds the span entry to the table. doc_text is sanitized by replacing the reserved
                // symbols by their entity name representations
                row.innerHTML += `
                <td>
                    <div class='sa-table-controls'>
                    <button data-control='visibility' style='width:1em'><svg style='display:block;margin:0.2em auto;' xmlns="http://www.w3.org/2000/svg" viewBox="0 0 576 512"><path d="M572.52 241.4C518.29 135.59 410.93 64 288 64S57.68 135.64 3.48 241.41a32.35 32.35 0 0 0 0 29.19C57.71 376.41 165.07 448 288 448s230.32-71.64 284.52-177.41a32.35 32.35 0 0 0 0-29.19zM288 400a144 144 0 1 1 144-144 143.93 143.93 0 0 1-144 144zm0-240a95.31 95.31 0 0 0-25.31 3.79 47.85 47.85 0 0 1-66.9 66.9A95.78 95.78 0 1 0 288 160z"/></svg></button>
                    <button data-control='highlight' style='width:1em'><svg style='display:block;margin:0.2em auto;' xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><path d="M256 160c-52.9 0-96 43.1-96 96s43.1 96 96 96 96-43.1 96-96-43.1-96-96-96zm246.4 80.5l-94.7-47.3 33.5-100.4c4.5-13.6-8.4-26.5-21.9-21.9l-100.4 33.5-47.4-94.8c-6.4-12.8-24.6-12.8-31 0l-47.3 94.7L92.7 70.8c-13.6-4.5-26.5 8.4-21.9 21.9l33.5 100.4-94.7 47.4c-12.8 6.4-12.8 24.6 0 31l94.7 47.3-33.5 100.5c-4.5 13.6 8.4 26.5 21.9 21.9l100.4-33.5 47.3 94.7c6.4 12.8 24.6 12.8 31 0l47.3-94.7 100.4 33.5c13.6 4.5 26.5-8.4 21.9-21.9l-33.5-100.4 94.7-47.3c13-6.5 13-24.7.2-31.1zm-155.9 106c-49.9 49.9-131.1 49.9-181 0-49.9-49.9-49.9-131.1 0-181 49.9-49.9 131.1-49.9 181 0 49.9 49.9 49.9 131.1 0 181z"/></svg></button>
                    </div>
                </td>
                <td><b>${entry.id.toString()}</b></td>
                <td>${entry.begin}</td>
                <td>${entry.end}</td>
                ${(doc.doc_token_spans != undefined) ? `<td>${doc.doc_token_spans[entry.id][0]}</td><td>${doc.doc_token_spans[entry.id][1]}</td>` : ''}
                <td>${sanitize(doc_text.substring(entry.begin, entry.end))}</td>`

                tbody.appendChild(row)
            })
            table.appendChild(tbody)
            frag.appendChild(table)
        }

        // Render Text
        let highlight_regions = []
        for(let i = 0; i < entries.length; i++)
        {
            if(!entries[i].visible) continue
            if(entries[i].valid_sets.length > 0)
            {
                let span = entries[i].getSetSpan();
                let ids = [entries[i].id, ...entries[i].valid_sets.map(set => { return set.entry.id })]
                if(entries[i].isComplex()) {
                    highlight_regions.push({begin: span.begin, end: span.end, type: TYPE_COMPLEX, ids: ids})
                } else {
                    highlight_regions.push({begin: span.begin, end: span.end, type: TYPE_NESTED, ids: ids})
                }
                i = span.highest_id
            } else {
                highlight_regions.push({begin: entries[i].begin, end: entries[i].end, type: TYPE_SOLO, ids: [entries[i].id]})
            }
        }

        let paragraph = document.createElement("p")
        if(highlight_regions.length == 0) {
            paragraph.innerHTML = sanitize(doc_text)
        } else {
            let begin = 0
            highlight_regions.forEach(region => {
                paragraph.innerHTML += sanitize(doc_text.substring(begin, region.begin))

                let mark = document.createElement("mark")
                // The data-ids tag is a list of comma-separated reference IDs for matching Spans 
                mark.setAttribute("data-ids", "");
                if (region.type != TYPE_NESTED) {
                    region.ids.forEach(id => {
                        mark.setAttribute("data-ids", mark.getAttribute("data-ids") + `#${id},`)
                        if(doc.lookup_table[id].highlighted) mark.classList.add("highlighted")
                    })
                    mark.innerHTML = sanitize(doc_text.substring(region.begin, region.end))
                } else {
                    mark.setAttribute("data-ids", `#${region.ids[0]},`)
                    if(doc.lookup_table[region.ids[0]].highlighted) mark.classList.add("highlighted")
                    let nested_begin = region.begin
                    region.ids.slice(1).forEach(nested_id => {
                        let nested_region = doc.lookup_table[nested_id]
                        mark.innerHTML += sanitize(doc_text.substring(nested_begin, nested_region.begin))
                        let nested_mark = document.createElement("mark")
                        nested_mark.setAttribute("data-ids", `#${nested_id},`)
                        if(nested_region.highlighted) nested_mark.classList.add("highlighted")
                        nested_mark.innerHTML = sanitize(doc_text.substring(nested_region.begin, nested_region.end))
                        nested_begin = nested_region.end
                        mark.appendChild(nested_mark)
                    })
                    mark.innerHTML += sanitize(doc_text.substring(nested_begin, region.end))
                }

                if(region.type == TYPE_COMPLEX) {
                    let markTag = document.createElement("span")
                    markTag.textContent = "Set"
                    markTag.classList.add("mark-tag")
                    mark.classList.add("complex-set")
                    mark.appendChild(markTag)
                }

                begin = region.end
                paragraph.appendChild(mark)
            })
            paragraph.innerHTML += sanitize(doc_text.substring(highlight_regions[highlight_regions.length - 1].end, doc_text.length))
        }
        
        frag.appendChild(paragraph)

        return frag
    }

    /** Attach hover and click events to a document render via event delegation */
    function attachDocumentEvents(fragment, doc_object, source_spanarray) {
        const doc_table_body = fragment.querySelector("table>tbody")
        const doc_text = fragment.querySelector("p")

        // Hover highlight events

        doc_table_body.addEventListener("pointerenter", (event) => {
            if(event.target.nodeName == "TR") {
                event.target.classList.add("hover")
                const span_id = event.target.getAttribute("data-id")
                const marks = doc_text.querySelectorAll("mark[data-ids]")
                Array.from(marks)
                    .filter(mark => {
                        return mark.getAttribute("data-ids").includes(`#${span_id},`)
                    })
                    .forEach(related_mark => {
                        related_mark.classList.add("hover")
                    })
            }
        }, true)

        doc_table_body.addEventListener("pointerleave", (event) => {
            if(event.target.nodeName == "TR") {
                event.target.classList.remove("hover")
                const span_id = event.target.getAttribute("data-id")
                const marks = doc_text.querySelectorAll("mark[data-ids]")
                Array.from(marks)
                    .filter(mark => {
                        return mark.getAttribute("data-ids").includes(`#${span_id},`)
                    })
                    .forEach(related_mark => {
                        related_mark.classList.remove("hover")
                    })
            }
        }, true)

        doc_text.addEventListener("pointerenter", (event) => {
            if(event.target.nodeName == "MARK") {
                event.target.classList.add("hover")
                const ids = event.target.getAttribute("data-ids").split(",").slice(0, -1)
                Array.from(ids)
                    .map(id_tag => {
                        return id_tag.substring(1)
                    })
                    .forEach(id => {
                        const entry = doc_table_body.querySelector(`tr[data-id="${id}"]`)
                        entry.classList.add("hover")
                    })
            }
        }, true)

        doc_text.addEventListener("pointerleave", (event) => {
            if(event.target.nodeName == "MARK") {
                event.target.classList.remove("hover")
                const ids = event.target.getAttribute("data-ids").split(",").slice(0, -1)
                Array.from(ids)
                    .map(id_tag => {
                        return id_tag.substring(1)
                    })
                    .forEach(id => {
                        const entry = doc_table_body.querySelector(`tr[data-id="${id}"]`)
                        entry.classList.remove("hover")
                    })
            }
        }, true)

        // Click disable/enable events

        doc_table_body.addEventListener("click", (event) => {
            const closest_control_button = event.target.closest("button[data-control]")
            if(closest_control_button == undefined) return

            const closest_tr = event.target.closest("tr")
            if(closest_tr == undefined) return

            const matching_span = doc_object.lookup_table[closest_tr.getAttribute("data-id")]
            if(matching_span == undefined) return

            switch(closest_control_button.getAttribute("data-control")) {
                case "visibility":
                    {
                        matching_span.visible = !matching_span.visible
                        source_spanarray.render()
                    }
                    break;
                case "highlight":
                    {
                        matching_span.highlighted = !matching_span.highlighted
                        source_spanarray.render()
                    }
                    break;
            }



        }, true)

        doc_text.addEventListener("click", (event) => {
            const closest_mark = event.target.closest("mark")
            if(closest_mark == undefined) return

            // Preprocess ID string into a list of IDs
            const ids = closest_mark.getAttribute("data-ids")
                .split(",")
                .slice(0, -1)
                .map(id => {
                    return id.substring(1)
                })
            
            // If any of the connected IDs are highlighted, we set all spans in the list to not highlighted.
            // Inversely, we want all spans highlighted if none were previously.

            const highlighted_entry = ids.find(id => {
                return doc_object.lookup_table[id].highlighted
            })

            const is_highlighted = (highlighted_entry != undefined)

            ids.forEach(id => {
                const span = doc_object.lookup_table[id]
                if(span != undefined) span.highlighted = !is_highlighted
            })

            source_spanarray.render()
        })
    }
} else {
    // SpanArray JS is already defined and not an outdated copy
    // Replace global SpanArray CSS with latest copy IFF global stylesheet is undefined
    
    if(local_stylesheet != undefined) {
        if(global_stylesheet == undefined) {
            document.head.appendChild(local_stylesheet)
        } else {
            document.currentScript.parentElement.removeChild(local_stylesheet)
        }
    }       
}