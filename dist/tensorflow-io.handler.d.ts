import * as tf from '@tensorflow/tfjs';
export declare class TensorflowIOHandler implements tf.io.IOHandler {
    json?: string | undefined;
    constructor(json?: string | undefined);
    saveModel(model: tf.Sequential): Promise<string>;
    encodeWeights(tensors: tf.Tensor<tf.Rank>[]): Promise<string>;
    correctArtifact(modelArtifact: tf.io.ModelArtifacts): tf.io.ModelArtifacts;
    get save(): tf.io.SaveHandler;
    get load(): tf.io.LoadHandler;
    saveWeights(weights: ArrayBuffer): Promise<string>;
    loadWeights(weights: string): Promise<ArrayBuffer>;
}
