var activeIndividuals = {};
var individualDivContainer = [];
var individualDivUpdateFunction = [];
var windowID = 0;
const graphNames = ["survival", "reproduction", "morphology"];


async function updateIndividualsButtons() {

    const container = document.getElementById('individual-selector');
    var keys = await eel.get_individuals_ID_from_python()();
    container.innerHTML = "";

    for (let key in keys) {
        let button = document.createElement('button');
        button.textContent = key;
        button.style.backgroundColor = getColorByKey(key);
        button.className = "individual-btn";

        button.addEventListener('click', function() {
            updateActiveIndividualStatsList(key);
            if (!activeStats.includes(updateAllIndividuals)) {
              activeStats.push(updateAllIndividuals);
            }
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

}

function addIndividualStats(index) {

    let [container, allChartComponents] = createDraggableDiv(index, activeIndividuals[index][0]);
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
    
    let scores = {};
    try {
        scores = await eel.get_all_stats_from_python(parseInt(activeIndividuals[windID][0]))();
    } catch (error) {
        console.error('Error parsing JSON:', error);
        console.log(scores);
    }

    if (scores.length === 0) { return; }

    // Get all the bar chart SVG elements inside the div
    for (let name of graphNames) {
        if (activeIndividuals[windID] != undefined) { 
            activeIndividuals[windID][2][name][2](activeIndividuals[windID][0], scores[name]);
        }
    }

    
}












function createDraggableDiv(windID, id) {
  // Create a new div element
  var div = document.createElement('div');

  div.setAttribute('data-custom', id);
  div.className = "individual-stat-window window"

  // Set some styles for the div
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
    var initialLeft = startX; // parseFloat(div.style.left);
    var initialTop = startY; // parseFloat(div.style.top);

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
        
    };
    
    div.appendChild(closeIndividualStatsButton);

    let allChartComponents = {};

    for (let name of graphNames) {
        let chartComponents = createBarChart("indiv-" + name + '-' + id);

        let titleContainer = document.createElement("div");
        titleContainer.className = "kernel-parameter-title parameter-title";
        div.appendChild(titleContainer);
        let newLbl = document.createElement("label");
        newLbl.textContent = name.toUpperCase();
        titleContainer.appendChild(newLbl);

        div.appendChild(chartComponents[0]);
        allChartComponents[name] = chartComponents
    }


    // Return the created div
    return [div, allChartComponents];
}


function closeIndivWindow(windID) {

    if (!activeIndividuals.hasOwnProperty(windID)) {
        return
    }

    for (let name in activeIndividuals[windID][2]) {

        delete activeIndividuals[windID][2][name];
    }

    activeIndividuals[windID][1].remove();
    
    delete activeIndividuals[windID];
    windowID = windID;
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
    .text("Time Units");

  chart.append("text")
    .attr("class", "axis-label")
    .attr("x", -140)
    .attr("y", -40)
    .attr("transform", "rotate(-90)")
    .text("Score");

  // Function to update the graph with new data
  function updateGraph(color, newData) {
    // Add the new data to the dataset
    dataset.push(newData);

    // Limit the dataset to 10 bars
    if (dataset.length > 100) {
      dataset.shift(); // Remove the oldest data point
    }

    // Update the scales
    xScale.domain(d3.range(dataset.length));
    yScale.domain([-1, 1]);

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

    // Exit
    bars.exit().remove();
  }

  // Return the created chart and the update function
  return [svg.node(), chart, updateGraph];
}