//import { Tensor, InferenceSession } from "onnxjs";

async function load_model() {
  const session = new onnx.InferenceSession({ backendHint: 'webgl' });
  //const url = "./actor_critic.onnx";
  const url = "input_params/actor_critic.onnx";
  await session.loadModel(url);
  console.log('Model loaded');
}

async function load_network() {
  await shapefile.open("./input_params/downtown_network/downtown_network.shp")
    .then(source => source.read()
      .then(function log(result) {
        if (result.done) return;
        console.log(result.value);
        return source.read().then(log);
      }))
    .catch(error => console.error(error.stack));
}

load_model()
load_network()
