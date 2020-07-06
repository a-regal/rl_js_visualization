const svg = d3.select("svg")
const myProjection = d3.geoNaturalEarth1()
const path = d3.geoPath().projection(myProjection)
const graticule = d3.geoGraticule()

function drawMap(err, world) {
  if (err) throw err
  svg.append("path")
    .datum(graticule)
    .attr("class", "graticule")
    .attr("d", path);
  svg.append("path")
    .datum(graticule.outline)
    .attr("class", "foreground")
    .attr("d", path);
  svg.append("g")
    .selectAll("path")
    .data(topojson.feature(world, world.objects.countries).features)
    .enter().append("path")
    .attr("d", path);
}

d3.json("https://unpkg.com/world-atlas@1.1.4/world/110m.json", drawMap)
