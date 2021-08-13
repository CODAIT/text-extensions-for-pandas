// Script to convert the parent widget's table from a widget structure to an HTML table wrapper for height preservation
{
    function widgetToTable(root_table_element) {
        // Limit queries to the table for scope and speed

        let table = document.createElement("table")

        // Pop the first children off tp--dfwidget--table to get the row headers
        let index_column = root_table_element.children[0]
        let header_row = document.createElement("tr")
        let header_cell
        for(let column of root_table_element.children)
        {
            header_cell = document.createElement("th")
            header_cell.appendChild(column.children[0])
            header_row.appendChild(header_cell)
        }
        table.appendChild(header_row)
        
        // Loop over children of tp--dfwidget--table and pop the first children off each into a row until all children expired
        let row;
        while(index_column.children.length > 0) {
            row = document.createElement("tr")
            for(let column of root_table_element.children)
            {
                cell = document.createElement("td")
                cell.appendChild(column.children[0])
                row.appendChild(cell)
            }
            table.appendChild(row)
        }

        root_table_element.appendChild(table);
    }

    let currentScript = document.currentScript

    let tryCount = 0;

    // Try to attach to the root table element on run
    let convertTimeout = setInterval(() => {
        tryCount += 1;

        if (tryCount > 1000) {
            clearInterval(convertTimeout);
            return;
        }

        let widget_root = currentScript.closest("div.tep--dfwidget--output")
        if (widget_root == null) return

        let root_table_element = widget_root.querySelector("div.tep--dfwidget--table")
        if (root_table_element == null) return

        widgetToTable(root_table_element)
        clearInterval(convertTimeout)

    }, 100)
}