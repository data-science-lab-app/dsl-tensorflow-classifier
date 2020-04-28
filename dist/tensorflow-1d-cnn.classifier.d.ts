import { AlgorithmPlugin, PluginOptions, PluginInputs, Option, RecorderService, PluginData, PluginDataInput } from 'data-science-lab-core';
import * as tf from '@tensorflow/tfjs';
interface Tensorflow1dCnnClassifierInput {
    inputData: number[][];
    inputLabels: number[][];
    labels: number[];
    activation: string;
    trainX: tf.Tensor2D;
    trainLabels: tf.Tensor2D;
    model: tf.Sequential;
    batchSize: number;
}
export declare class Tensorflow1dCnnClassifier extends AlgorithmPlugin {
    options: Tensorflow1dCnnClassifierPluginOptions;
    inputs: Tensorflow1dCnnClassifierPluginInputs;
    data: Tensorflow1dCnnClassifierInput;
    recorder?: RecorderService;
    constructor();
    setInput(examples: number[][]): void;
    setOutput(examples: number[][]): void;
    setRecorderService(recorder: RecorderService): void;
    getInputs(): Tensorflow1dCnnClassifierPluginInputs;
    getOptions(): Tensorflow1dCnnClassifierPluginOptions;
    finishTraining(): boolean;
    export(minimal: boolean): Promise<string>;
    import(json: string, minimal: boolean): Promise<Tensorflow1dCnnClassifier>;
    test(argument: {
        [id: string]: any[];
    }): {
        [id: string]: any[];
    };
    initialize(): void;
    step(): Promise<void>;
    autoDetect(): number[];
    setActivation(activation: string): void;
    setLabels(labels: number[]): void;
    setBatchSize(batch: number): void;
    getTestingInputs(): {
        input: PluginDataInput[];
        output?: PluginDataInput[];
    };
}
declare class Tensorflow1dCnnClassifierPluginOptions extends PluginOptions {
    classifier: Tensorflow1dCnnClassifier;
    state: number;
    labels: number[];
    constructor(classifier: Tensorflow1dCnnClassifier);
    submit(inputs: {
        [id: string]: any;
    }): void;
    executeCommand(id: string): Promise<void>;
    options(): Option[];
    noMore(): boolean;
}
declare class Tensorflow1dCnnClassifierPluginInputs extends PluginInputs {
    classifier: Tensorflow1dCnnClassifier;
    constructor(classifier: Tensorflow1dCnnClassifier);
    inputs(): PluginDataInput[];
    submit(inputs: {
        [id: string]: PluginData;
    }): void;
}
export {};
