//import { Tensor, InferenceSession } from "onnxjs";

async function load_model() {
  const session = new onnx.InferenceSession();
  const url = "input_params/actor_critic.onnx";
  await session.loadModel(url);
  console.log('Model loaded');
}

load_model()
