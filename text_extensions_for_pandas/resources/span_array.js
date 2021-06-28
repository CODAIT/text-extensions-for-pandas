// Increment the version to invalidate the cached script
const VERSION = 0.62

if(!window.SpanArray || window.SpanArray.VERSION < VERSION) {

    // Sets up the SpanArray global namespace
    window.SpanArray = {}
    window.SpanArray.VERSION = VERSION

    window.SpanArray.TYPE_OVERLAP = 0;
    window.SpanArray.TYPE_NESTED = 1;
    window.SpanArray.TYPE_COMPLEX = 2;
    window.SpanArray.TYPE_SOLO = 3;

    TYPE_OVERLAP = window.SpanArray.TYPE_OVERLAP;
    TYPE_NESTED = window.SpanArray.TYPE_NESTED;
    TYPE_COMPLEX = window.SpanArray.TYPE_COMPLEX;
    TYPE_SOLO = window.SpanArray.TYPE_SOLO;

    function sanitize(input) {
        let out = input.slice();
        out = out.replace("&", "&amp;")
        out = out.replace("<", "&lt;")
        out = out.replace(">", "&gt;")
        out = out.replace("$", "&#36;")
        out = out.replace("\"", "&quot;")
        return out;
    }

    /** Models an instance of a SpanArray, with document-separated spans and text
     * NOTE: Using docs instead of documents to avoid unintentionally manipulating the global 'document' object.
    */
    class SpanArray {
        constructor(docs, show_offsets, script_context) {
            this.docs = docs
            this.show_offsets = show_offsets
            this.script_context = script_context
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
                doc_container.appendChild(getDocumentFragment(doc.doc_text, doc.doc_spans, this.show_offsets))
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
            let id = 0
            
            spanArray.sort((a, b) => {
                if(a[0] < b[0]) {
                    return a
                } else if(a[0] == b[0] && a[1] >= b[1]) {
                    return a
                } else {
                    return b
                }
            })
            .forEach(span => {
                entries.push(new Span(id, span[0], span[1]))
                id += 1
            })

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
    function getDocumentFragment(doc_text, entries, show_offsets) {

        let frag = document.createDocumentFragment()
        
        // Render Table
        if(show_offsets) {
            let table = document.createElement("table")
            table.innerHTML = `
            <thead>
            <tr>
                <th></th>
                <th>begin</th>
                <th>end</th>
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

                // Adds the span entry to the table. doc_text is sanitized by replacing the reserved
                // symbols by their entity name representations
                row.innerHTML += `
                <td><b>${entry.id.toString()}</b></td>
                <td>${entry.begin}</td>
                <td>${entry.end}</td>
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
            paragraph.textContent = doc_text
        } else {
            let begin = 0
            highlight_regions.forEach(region => {
                paragraph.innerHTML += sanitize(doc_text.substring(begin, region.begin))

                let mark = document.createElement("mark")
                // The data-ids tag is a list of comma-separated reference IDs for matching Spans 
                mark.setAttribute("data-ids", "");
                if (region.type != TYPE_NESTED) {
                    region.ids.forEach(id => {
                        mark.setAttribute("data-ids", mark.getAttribute("data-ids") + `${id},`)
                    })
                    mark.textContent = doc_text.substring(region.begin, region.end)
                } else {
                    mark.setAttribute("data-ids", `${region.ids[0]},`)
                    let nested_begin = region.begin
                    region.ids.slice(1).forEach(nested_id => {
                        let nested_region = entries.find(entry => entry.id == nested_id)
                        mark.innerHTML += sanitize(doc_text.substring(nested_begin, nested_region.begin))
                        let nested_mark = document.createElement("mark")
                        nested_mark.setAttribute("data-ids", `${nested_id},`)
                        nested_mark.textContent = doc_text.substring(nested_region.begin, nested_region.end)
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
            paragraph.innerHTML += sanitize(doc_text.substring(entries[entries.length - 1].end, doc_text.length))
        }
        
        frag.appendChild(paragraph)

        return frag
    }
}