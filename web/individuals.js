var activeIndividuals = {};
var individualDivContainer = [];
var individualDivUpdateFunction = [];
var windowID = 0;
const graphNames = ["survival", "reproduction", "morphology"];


async function updateIndividualsButtons() {

    const container = document.getElementById('individual-selector');
    var dictionary = await eel.getIndividualsFromPython()();
    // console.log(dictionary);
    container.innerHTML = "";

    for (let key in dictionary) {
        let button = document.createElement('button');
        button.textContent = key;
        button.style.backgroundColor = getColorByKey(key);
        button.className = "individual-btn";

        button.addEventListener('click', function() {
            updateActiveIndividualStatsList(key);
            // console.log("You pressed key: " + key);
        });
        
        container.appendChild(button);
    }
}

async function updateActiveIndividualStatsList(id) {

    for (let windID in activeIndividuals) {
        if (activeIndividuals[windID][0] === id) {
            return;
        }
    }
    closeIndivWindow(windowID);
    activeIndividuals[windowID] = [id];
    addIndividualStats(windowID);
    windowID = (windowID + 1) % 2;
    // console.log(windowID);
    
    // console.log(activeIndividuals);
    

    // updateIndividualsStatWindows();

}

function updateIndividualsStatWindows() {

    // individualDivContainer = [];

    for (let i = 0; i < activeIndividuals.length; i++) {
        individualDivContainer.push(createDraggableDiv(i, activeIndividuals[i]));
        
        individualDivContainer[i].style.display = "flex";
        document.body.appendChild(individualDivContainer[i]);
    }

}

function addIndividualStats(index) {

    // let openedWindows = [];

    // for (let idc in individualDivContainer) {
    //     openedWindows.push(idc.getAttribute("data-custom"));
    //     updateIndividual(i, activeIndividuals[i]);
    // }

    // for (let i = 0; i < activeIndividuals.length; i++) {
    //     if (openedWindows.includes(activeIndividuals[i])) {
    //         continue;
    //     }
    //     individualDivContainer.push(createDraggableDiv(i, activeIndividuals[i]));
    // }

    let [container, allChartComponents] = createDraggableDiv(index, activeIndividuals[index][0]);
    // individualDivContainer.push(chartComponents[0]);
    activeIndividuals[index].push(container);
    activeIndividuals[index].push(allChartComponents);
    activeIndividuals[index][1].style.display = "inline";
    document.body.appendChild(activeIndividuals[index][1]);
}

async function updateAllIndividuals() {
    for (let windID in activeIndividuals) {
        await updateIndividual(windID);
    }
}

async function updateIndividual(windID) {

    // let key = activeIndividuals[key][1].getAttribute("data-custom");

    let scores = await eel.getAllStatsFromPython(parseInt(activeIndividuals[windID][0]))();

    if (scores.length === 0) { return; }

    // Get all the bar chart SVG elements inside the div
    // const barCharts = individualDivContainer[index].querySelectorAll('g');

    // console.log(scores, barCharts[0]);

    // console.log(activeIndividuals[windID][2][0], scores["survival"]);
    for (let name of graphNames) {
        activeIndividuals[windID][2][name][2](activeIndividuals[windID][0], scores[name]);
    }
    // updateGraph(barCharts[0], [scores["survival"]]);



    // console.log(scores);

    // for (let idc in individualDivContainer) {
    //     if (idc.getAttribute("data-custom") === id) {
    //     updateIndividual(i, activeIndividuals[i]);
    // }



    // if (drg_wind.getAttribute("data-custom") != String(id)) {
    //     // Clear existing stats
    //     pass
    // }

    

    
}












function createDraggableDiv(windID, id) {
  // Create a new div element
  var div = document.createElement('div');

//   div.id = "individual-stat-window-" + String(index);
  div.setAttribute('data-custom', id);
  div.className = "individual-stat-window window"

  // Set some styles for the div
//   div.style.width = '200px';
//   div.style.height = '200px';
  div.style.position = 'absolute';
  div.style.left = '50%';
  div.style.top = '50%';
  div.style.cursor = 'move';

  // Add event listeners for dragging functionality
  div.addEventListener('mousedown', startDrag);
  div.addEventListener('touchstart', startDrag);

  // Function to handle drag start event
  function startDrag(e) {
    // Get initial mouse position
    var startX = e.clientX || e.touches[0].clientX;
    var startY = e.clientY || e.touches[0].clientY;

    // Get initial div position
    var initialLeft = parseFloat(div.style.left);
    var initialTop = parseFloat(div.style.top);

    // Function to handle drag move event
    function moveDiv(e) {
      e.preventDefault();

      // Calculate the distance moved by the mouse
      var deltaX = (e.clientX || e.touches[0].clientX) - startX;
      var deltaY = (e.clientY || e.touches[0].clientY) - startY;

      // Update the div position
      div.style.left = initialLeft + deltaX + 'px';
      div.style.top = initialTop + deltaY + 'px';
    }

    // Function to handle drag end event
    function stopDrag() {
      // Remove the event listeners for move and end events
      document.removeEventListener('mousemove', moveDiv);
      document.removeEventListener('touchmove', moveDiv);
      document.removeEventListener('mouseup', stopDrag);
      document.removeEventListener('touchend', stopDrag);
    }

    // Add event listeners for move and end events
    document.addEventListener('mousemove', moveDiv);
    document.addEventListener('touchmove', moveDiv);
    document.addEventListener('mouseup', stopDrag);
    document.addEventListener('touchend', stopDrag);
  }

    let closeIndividualStatsButton = document.createElement("button");
    closeIndividualStatsButton.type = "button";
    closeIndividualStatsButton.innerHTML = "<span class='material-symbols-outlined'> close </span>";
    closeIndividualStatsButton.onclick = function() { 

        closeIndivWindow(windID);
        
        // for (let name in activeIndividuals[windID][2]) {
        //     // indiv[2][name].remove();

        //     delete activeIndividuals[windID][2][name];
        // }
        // // chartComponents[1].remove();
        // div.remove();
        
        // delete activeIndividuals[windID];
        // windowID = windID;
        // activeIndividuals
    };
    
    div.appendChild(closeIndividualStatsButton);

    let allChartComponents = {};

    for (let name of graphNames) {
        let chartComponents = createBarChart("indiv-" + name + '-' + id);

        div.appendChild(chartComponents[0]);
        allChartComponents[name] = chartComponents
        // console.log(name)

    }
    // let [survivalChartNode, survivalChart, survivalUpdateFunc] = createBarChart("indiv-surv-" + id);
    // let [reproductionChartNode, reproductionChart, reproductionUpdateFunc] = createBarChart("indiv-reprod-" + id);
    // let [morphologyChartNode, morphologyChart, morphologyUpdateFunc] = createBarChart("indiv-morph-" + id);
    // let reproductionBarChart = createBarChart("indiv-reprod-" + id, []);
    // let morphologyBarChart = createBarChart("indiv-morph-" + id, []);
    
    // div.appendChild(reproductionBarChart);
    // div.appendChild(morphologyBarChart);


    // Return the created div
    return [div, allChartComponents];
}


function closeIndivWindow(windID) {

    if (!activeIndividuals.hasOwnProperty(windID)) {
        return
    }

    for (let name in activeIndividuals[windID][2]) {
        // indiv[2][name].remove();

        delete activeIndividuals[windID][2][name];
    }
    // chartComponents[1].remove();
    activeIndividuals[windID][1].remove();
    
    delete activeIndividuals[windID];
    windowID = windID;
}



function createBarChart2(id) {
    // Initialize an empty dataset
    let dataset = [];

    // Create the initial empty bar graph
    const svg = d3.create("svg")
        .attr("class", "individual-chart-container");

    const chart = svg.append("g")
        .attr("id", id)
        .attr("class", "bar-chart individual-chart")
        .attr("transform", "translate(50, 50)");

    // Set up the scales
    const xScale = d3.scaleBand()
        .domain(d3.range(10))
        .range([0, 400])
        .padding(0.1);

    const yScale = d3.scaleLinear()
        .domain([0, 100])
        .range([200, 0]);

    // Create the axis
    const xAxis = d3.axisBottom(xScale).tickSize(0).tickPadding(10);
    const yAxis = d3.axisLeft(yScale).tickSize(0).tickPadding(10);

    chart.append("g")
        .attr("transform", "translate(0, 200)")
        .call(xAxis)
        .selectAll("text")
        .style("fill", "black");

    chart.append("g")
        .call(yAxis)
        .selectAll("text")
        .style("fill", "black");

    // Add axis labels
    chart.append("text")
        .attr("class", "axis-label")
        .attr("x", 220)
        .attr("y", 250)
        .style("fill", "black")
        .text("Bars");

    chart.append("text")
        .attr("class", "axis-label")
        .attr("x", -140)
        .attr("y", -40)
        .attr("transform", "rotate(-90)")
        .style("fill", "black")
        .text("Height");

    // Update the graph with new data
    function updateGraph(newData) {
        // Generate random data for the new bar
        // const newData = Math.random() * 100;

        // Add the new data to the dataset
        dataset.push(newData);

        // Limit the dataset to 10 bars
        if (dataset.length > 10) {
            dataset.shift(); // Remove the oldest data point
        }

        // Update the scales
        xScale.domain(d3.range(dataset.length));
        yScale.domain([0, d3.max(dataset)]);

        // Update the vertical bar graph
        const bars = chart.selectAll("rect")
        .data(dataset);

        // Enter
        bars.enter()
        .append("rect")
        .attr("class", "bar")
        .attr("x", (d, i) => xScale(i))
        .attr("y", (d) => yScale(d))
        .attr("width", xScale.bandwidth())
        .attr("height", (d) => 200 - yScale(d));

        // Update
        bars.attr("x", (d, i) => xScale(i))
        .attr("y", (d) => yScale(d))
        .attr("width", xScale.bandwidth())
        .attr("height", (d) => 200 - yScale(d));

        // Exit
        bars.exit().remove();
    }

    // Update the graph every 1 second (1000 milliseconds)
    // setInterval(updateGraph, 1000);

    // Return the created chart
    return svg.node();
}


function createBarChart(id) {
  // Initialize an empty dataset
  let dataset = [];

    // Create the initial empty bar graph
    const svg = d3.create("svg")
        .attr("class", "individual-chart-container");

    const chart = svg.append("g")
        .attr("id", id)
        .attr("class", "bar-chart individual-chart")
        .attr("transform", "translate(50, 50)");

  // Set up the scales
  const xScale = d3.scaleBand()
    .domain(d3.range(10))
    .range([0, 400])
    .padding(0.1);

  const yScale = d3.scaleLinear()
    .domain([-1, 1]) // Set the y-axis domain to be between 0 and 1
    .range([200, 0]);

  // Create the axis
  const xAxis = d3.axisBottom(xScale);
  const yAxis = d3.axisLeft(yScale);

  chart.append("g")
    .attr("transform", "translate(0, 200)")
    .call(xAxis);

  chart.append("g")
    .call(yAxis);

  // Add axis labels
  chart.append("text")
    .attr("class", "axis-label")
    .attr("x", 220)
    .attr("y", 250)
    .text("Bars");

  chart.append("text")
    .attr("class", "axis-label")
    .attr("x", -140)
    .attr("y", -40)
    .attr("transform", "rotate(-90)")
    .text("Height");

  // Function to update the graph with new data
  function updateGraph(color, newData) {
    // Add the new data to the dataset
    dataset.push(newData);

    // Limit the dataset to 10 bars
    if (dataset.length > 10) {
      dataset.shift(); // Remove the oldest data point
    }

    // Update the scales
    xScale.domain(d3.range(dataset.length));
    yScale.domain([0, d3.max(dataset)]);

    // Update the vertical bar graph
    const bars = chart.selectAll("rect")
      .data(dataset);

    // Enter
    bars.enter()
      .append("rect")
      .attr("class", "bar")
      .attr("x", (d, i) => xScale(i))
      .attr("y", (d) => yScale(d))
      .attr("width", xScale.bandwidth())
      .attr("height", (d) => 200 - yScale(d))
      .attr("fill", getColorByKey(color));

    // Update
    bars.attr("x", (d, i) => xScale(i))
      .attr("y", (d) => yScale(d))
      .attr("width", xScale.bandwidth())
      .attr("height", (d) => 200 - yScale(d));
    //   .attr("fill", getColorByKey(color));

    // Exit
    bars.exit().remove();
  }

  // Return the created chart and the update function
  return [svg.node(), chart, updateGraph];
}

// Example usage:
// const { chart, updateGraph } = createBarChart();
// // Use the updateGraph function to update the chart with new data
// updateGraph(50);
// updateGraph(75);
// updateGraph(30);
// ... and so on
