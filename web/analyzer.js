


// Define width and height of chart
const margin = { top: 20, right: 20, bottom: 50, left: 40 };
const width = 500 - margin.left - margin.right;
const height = 200 - margin.top - margin.bottom;

// Create scale functions for x and y axes
const xScale_all = [];
const yScale_all = [];
const svg_all = [];
const xAxis_all = [];
const yAxis_all = [];

const chart_ids = [
    "#global-survival",
    "#global-reproduction",
    "#global-morphology"
]

for (let i = 0; i < chart_ids.length; i++) {
  // Create x scale
  const xScale = d3.scaleBand()
    .range([0, width])
    .paddingInner(0.1);
  xScale_all.push(xScale);

  // Create y scale
  const yScale = d3.scaleLinear()
    .domain([-1, 1])
    .range([height, 0]);
  yScale_all.push(yScale);

  // Create SVG element and append to chart container
  const svg = d3.select(chart_ids[i])
    .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
  svg_all.push(svg);

  // Create x axis and append to chart
  const xAxis = d3.axisBottom()
    .scale(xScale);
  xAxis_all.push(xAxis);
  svg.append("g")
    .attr("class", "x axis")
    .attr("transform", "translate(0," + height + ")")
    .call(xAxis);

  // Create y axis and append to chart
  const yAxis = d3.axisLeft()
    .scale(yScale);
  yAxis_all.push(yAxis);
  svg.append("g")
    .attr("class", "y axis")
    .call(yAxis);
}



// Function to generate random data
function generateData() {
  const numBars = Math.floor(Math.random() * 10) + 1;
  const data = {};
  for (let i = 0; i < numBars; i++) {
    data[String.fromCharCode(65 + i)] = Math.floor(Math.random() * 100);
  }
  return data;
}

// Define a color function that maps keys to colors
function getColorByKey(id) {
    // Define your color mapping logic here
    // For example, you can use a switch statement or an object lookup
    // to assign a specific color to each key
    let inputNum = Math.floor((id + 50) * 103);
    let r = (inputNum * 937) % 128 + 64;
    let g = (inputNum * 961) % 128 + 96;
    let b = (inputNum * 989) % 128 + 100;
    // console.log(number, red, green, blue);
    return `rgb(${r}, ${g}, ${b})`;
}

// Update chart with new data
function updateChart(data, chart_id) {
  // Update x scale domain with new labels
  xScale_all[chart_id].domain(Object.keys(data));

  // Update bars with new data
  const bars = svg_all[chart_id].selectAll("rect")
    .data(Object.entries(data), d => d[0]);

  bars.enter()
    .append("rect")
    .merge(bars)
    .attr("x", d => xScale_all[chart_id](d[0]))
    .attr("y", d => yScale_all[chart_id](d[1]))
    .attr("width", xScale_all[chart_id].bandwidth())
    .attr("height", d => height - yScale_all[chart_id](d[1]))
    .attr("fill", d => getColorByKey(d[0])); // Set the fill color based on the key

  bars.exit()
    .remove();

  // Add numeric labels for each sample below their respective bar in the horizontal axis
  svg_all[chart_id].selectAll(".label")
    .data(Object.entries(data), d => d[0])
    .enter()
    .append("text")
    .attr("class", "label")
    .merge(svg_all[chart_id].selectAll(".label"))
    .attr("x", d => xScale_all[chart_id](d[0]) + xScale_all[chart_id].bandwidth() / 2)
    .attr("y", height + 20)
    .attr("text-anchor", "middle")
    .text(d => d[0]) // Use the key as the label
    .style("fill", "white");

// d3.selectAll(".bar-label")

  svg_all[chart_id].selectAll(".label")
    .data(Object.entries(data), d => d[0])
    .exit()
    .remove();

  // Update y axis domain with a maximum value of 1.0
  yScale_all[chart_id].domain([-1, 1.0]);

  svg_all[chart_id].select(".y.axis")
    .call(yAxis_all[chart_id]);
}




async function getNewStats(chart_id) {
    
    // let data = await eel.getGlobalReproductionStats()();

    var data;

    switch (chart_id) {
        case 0:
            data = await eel.getGlobalSurvivalStats()();
            break;
        case 1:
            data = await eel.getGlobalReproductionStats()();
            break;
        default:
            data = await eel.getGlobalMorphologyStats()();
    }

    updateChart(data, chart_id);
    
}
