//import { Tensor, InferenceSession } from "onnxjs";

async function load_model() {
  const session = new onnx.InferenceSession({ backendHint: 'webgl' });
  //const url = "./actor_critic.onnx";
  const url = "input_params/actor_critic.onnx";
  await session.loadModel(url);
  console.log('Model loaded');
}

load_model()
