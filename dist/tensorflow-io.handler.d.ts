import * as tf from '@tensorflow/tfjs';
export declare class TensorflowIOHandler implements tf.io.IOHandler {
    json?: string | undefined;
    constructor(json?: string | undefined);
    get save(): tf.io.SaveHandler;
    get load(): tf.io.LoadHandler;
    saveWeights(weights: ArrayBuffer): Promise<string>;
    loadWeights(weights: string): Promise<ArrayBuffer>;
}
