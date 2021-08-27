{
    let widget_root;
    let document_root;

    function enterEventHandler(event) {

        document_root = widget_root.querySelector(".tep--spanvis")
        if (!document_root) return;

        // When a hover target is part of a tep--spanvis--span, highlight all table entries that share matching IDs
        let closest_span = event.target.closest(".tep--spanvis--span")
        if(closest_span != undefined) {
            closest_span.classList.add("tep--hover--highlight")

            // Get all cells from the corresponding rows and add the proper class
            closest_span.getAttribute("data-ids").split(" ").forEach(id => {
                let query = `.tep--dfwidget--row-${id}`;
                widget_root.querySelectorAll(query).forEach(row => {
                    row.classList.add("tep--hover--highlight")
                })
            })
            return // Early exit
        }

        // If the closest hover target is a 'row' in the table
        let closest_table_cell = event.target.closest("[class*=tep--dfwidget--row]")
        if(closest_table_cell != undefined) {
            // Extract the row class
            let row_class = closest_table_cell.classList.value.split(" ").filter(cl => cl.includes('tep--dfwidget--row-'));
            if(row_class.length == 1)
            {
                row_class = row_class[0];
                // Query all cells from the same row
                widget_root.querySelectorAll(`.${row_class}`).forEach(element => {
                    element.classList.add("tep--hover--highlight")
                })
                // Query all spans associated to the row
                let row = row_class.split("tep--dfwidget--row-")[1];
                if(row != undefined)
                {
                    widget_root.querySelectorAll(`.tep--spanvis--span[data-ids~="${row}"]`).forEach(element => {
                        element.classList.add("tep--hover--highlight")
                    })
                }
            }
            return // Early exit
        }
    }

    function leaveEventHandler(event) {

        document_root = widget_root.querySelector(".tep--spanvis")
        if (!document_root) return;

        // when a hover target is part of a tep--spanvis--span, highlight all table entries that share matching IDs
        let closest_span = event.target.closest(".tep--spanvis--span")
        if(closest_span != undefined) {
            closest_span.classList.remove("tep--hover--highlight")

            // Get all cells from the corresponding rows and add the proper class
            closest_span.getAttribute("data-ids").split(" ").forEach(id => {
                let query = `.tep--dfwidget--row-${id}`;
                widget_root.querySelectorAll(query).forEach(row => {
                    row.classList.remove("tep--hover--highlight")
                })
            })
            return
        }

        // If the closest hover target is a 'row' in the table
        let closest_table_cell = event.target.closest("[class*=tep--dfwidget--row]")
        if(closest_table_cell != undefined) {
            // Extract the row class
            let row_class = closest_table_cell.classList.value.split(" ").filter(cl => cl.includes('tep--dfwidget--row-'));
            if(row_class.length == 1)
            {
                row_class = row_class[0];
                // Query all cells from the same row
                widget_root.querySelectorAll(`.${row_class}`).forEach(element => {
                    element.classList.remove("tep--hover--highlight")
                })
                // Query all spans associated to the row
                let row = row_class.split("tep--dfwidget--row-")[1];
                if(row != undefined)
                {
                    widget_root.querySelectorAll(`.tep--spanvis--span[data-ids~="${row}"]`).forEach(element => {
                        element.classList.remove("tep--hover--highlight")
                    })
                }
            }
            return // Early-ish exit
        }
    }

    function clickEventHandler(event) {

        document_root = widget_root.querySelector(".tep--spanvis")
        if (!document_root) return;

        // when a hover target is part of a tep--spanvis--span, highlight all table entries that share matching IDs
        let closest_span = event.target.closest(".tep--spanvis--span")
        if(closest_span != undefined) {
            let was_clicked = closest_span.classList.contains("tep--click--highlight")
            console.log("Click span", was_clicked);
            if (!was_clicked)
            {
                closest_span.classList.add("tep--click--highlight")
            }
            else 
            {
                closest_span.classList.remove("tep--click--highlight")
            }
            // Get all cells from the corresponding rows and add the proper class
            closest_span.getAttribute("data-ids").split(" ").forEach(id => {
                let query = `.tep--dfwidget--row-${id}`;
                widget_root.querySelectorAll(query).forEach(row => {
                    (!was_clicked) ? 
                        row.classList.add("tep--click--highlight") :
                        row.classList.remove("tep--click--highlight")
                })
            })
            return // Early exit
        }

    }

    (() => {
        let currentScript = document.currentScript;
        setTimeout(() => {
            // On load, navigate to widget
            widget_root = currentScript.closest("div.tep--dfwidget--output")
            if (widget_root == null) return

            document_root = widget_root.querySelector(".tep--spanvis")
    
            if(document_root) {
                // Attach event delegator to the root output widget
                widget_root.addEventListener("pointerenter", enterEventHandler, true);
                widget_root.addEventListener("pointerleave", leaveEventHandler, true);
                widget_root.addEventListener("click", clickEventHandler, true)
            }
        }, 100)
    })()
}