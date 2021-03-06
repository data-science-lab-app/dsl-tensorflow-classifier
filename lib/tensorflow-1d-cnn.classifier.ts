import { AlgorithmPlugin, PluginOptions, PluginInputs, Option, RecorderService, PluginData, PluginDataInput, NumberOption, ChoicesOption, CommandOption, TextOption } from 'data-science-lab-core';
import * as tf from '@tensorflow/tfjs';
import * as tfCore from '@tensorflow/tfjs-core';
import { TensorflowIOHandler } from './tensorflow-io.handler';
import { NamedTensorMap, tensor } from '@tensorflow/tfjs';
import { TypedArray } from '@tensorflow/tfjs-core/dist/types';
import { DTYPE_VALUE_SIZE_MAP } from '@tensorflow/tfjs-core/dist/io/types';

interface Tensorflow1dCnnClassifierInput {
    inputData: number[][];
    inputLabels: number[][];
    labels: number[];
    activation: string;
    trainX: tf.Tensor3D;
    trainLabels: tf.Tensor2D;
    model: tf.Sequential;
    batchSize: number;
}


interface Tensorflow1dCnnClassifierData {
    inputData: number[][];
    inputLabels: number[][];
    labels: number[];
    batchSize: number;
    model_config: string;
}

interface Tensorflow1dCnnClassifierMinimalData {
    labels: number[];
    model_config: string;
}

export class Tensorflow1dCnnClassifier extends AlgorithmPlugin {

    options: Tensorflow1dCnnClassifierPluginOptions;
    inputs: Tensorflow1dCnnClassifierPluginInputs;
    data: Tensorflow1dCnnClassifierInput;
    recorder?: RecorderService;

    constructor() {
        super();

        this.options = new Tensorflow1dCnnClassifierPluginOptions(this);
        this.inputs = new Tensorflow1dCnnClassifierPluginInputs(this);
        this.data = {} as Tensorflow1dCnnClassifierInput;
    }

    setInput(examples: number[][]) {
        this.data.inputData = examples;
    }

    setOutput(examples: number[][]) {
        this.data.inputLabels = examples;
    }

    setRecorderService(recorder: RecorderService) {
        this.recorder = recorder;
    }

    getInputs() {
        return this.inputs;
    }

    getOptions() {
        return this.options;
    }

    finishTraining() {
        return false;
    }

    async saveModel(model: tf.Sequential): Promise<string> {
        const artifact: tf.io.ModelArtifacts = {
            modelTopology: model.toJSON({}, false),
            format: 'layers-model',
            generatedBy: "TensorFlow.js tfjs-layers v1.7.3",
            convertedBy: null,
            userDefinedMetadata: model.getUserDefinedMetadata(),
            weightSpecs: model.getWeights().map((value) => {
                const variable = value as tf.Variable;
                return {
                    name: variable.name,
                    shape: variable.shape,
                    dtype: variable.dtype as any
                }
            })
        };
        model.getWeights()
        const json = {
            artifact,
            weights: await this.encodeWeights(model.getWeights())
        };
        return JSON.stringify(json);
    }

    async encodeWeights(tensors: tf.Tensor<tf.Rank>[]): Promise<string> {
        const data = await Promise.all(tensors.map(t => t.data()));
        let totalByteLength = 0;
        data.forEach(datum => {
            totalByteLength += datum.byteLength;
        });
        const y = new Uint8Array(totalByteLength);
        let offset = 0;
        data.forEach((x) => {
            y.set(new Uint8Array(x.buffer), offset);
            offset += x.byteLength;
        });

        const out = Buffer.from(y.buffer).toString('binary'); 
        return out;
    }

    async decodeWeights(weights: string, specs: tfCore.io.WeightsManifestEntry[]): Promise<tf.Tensor[]> {
        const temp = Buffer.from(weights, 'binary');
        const buffer = temp.buffer.slice(temp.byteOffset, temp.byteOffset + temp.byteLength);
        const out: tf.Tensor[] = [];
        let offset = 0;
        for (const spec of specs) {
            const dtype = spec.dtype;
            const shape = spec.shape;
            const size = tfCore.util.sizeFromShape(shape);
            let values: TypedArray | string[] | Uint8Array[];

            const dtypeFactor = DTYPE_VALUE_SIZE_MAP[dtype];
            const byteBuffer = buffer.slice(offset, offset + size * dtypeFactor);
            values = new Float32Array(byteBuffer);

            offset += size * dtypeFactor;

            out.push(tensor(values, shape, dtype));
        }
        return out;
    }


    // async export(minimal: boolean): Promise<string> {
    //     const handler = new TensorflowIOHandler();
    //     // const temp = this.decodeWeights('', []);
    //     // this.data.model.setWeights(temp);
    //     await this.data.model.save(handler);
    //     if (minimal) {
    //         const data: Tensorflow1dCnnClassifierMinimalData = {
    //             labels: this.data.labels,
    //             model_config: handler.json as string
    //         };
    //         return JSON.stringify(data);
    //     } else {
    //         const data: Tensorflow1dCnnClassifierData = {
    //             batchSize: this.data.batchSize,
    //             inputData: this.data.inputData,
    //             inputLabels: this.data.inputLabels,
    //             labels: this.data.labels,
    //             model_config: handler.json as string

    //         };
    //         return JSON.stringify(data);
    //     }
    // }
    async export(minimal: boolean): Promise<string> {
        const weights = await this.encodeWeights(this.data.model.getWeights());
        const specs = this.data.model.getWeights().map((value) => {
            const variable = value as tf.Variable;
            return {
                name: variable.name,
                shape: variable.shape,
                dtype: variable.dtype as any
            }
        });

        if (minimal) {
            const data = {
                labels: this.data.labels,
                weights,
                specs,
                activation: this.data.activation,
                inputShape: [this.data.inputData[0].length, 1]
            };
            return JSON.stringify(data);
        } else {
            const data = {
                labels: this.data.labels,
                weights,
                specs,
                activation: this.data.activation,
                inputShape: [this.data.inputData[0].length, 1],
                batchSize: this.data.batchSize,
                inputData: this.data.inputData,
                inputLabels: this.data.inputLabels,
            };
            return JSON.stringify(data);
        }
    }

    async import(json: string, minimal: boolean): Promise<Tensorflow1dCnnClassifier> {
        const data: any = JSON.parse(json);
        this.data.labels = data.labels;
        this.data.activation = data.activation;
        const shape = data.inputShape as number[];

        this.data.model = tf.sequential();

        this.data.model.add(tf.layers.conv1d({
            inputShape: shape,
            kernelSize: 5,
            filters: 8,
            strides: 1,
            activation: 'relu',
            kernelInitializer: 'varianceScaling'
        }));
        this.data.model.add(tf.layers.maxPooling1d({ poolSize: 2, strides: 2 }));

        this.data.model.add(tf.layers.conv1d({
            kernelSize: 5,
            filters: 16,
            strides: 1,
            activation: 'relu',
            kernelInitializer: 'varianceScaling'
        }));
        this.data.model.add(tf.layers.maxPooling1d({ poolSize: 2, strides: 2 }));

        this.data.model.add(tf.layers.flatten());
        this.data.model.add(tf.layers.dense({
            units: this.data.labels.length,
            kernelInitializer: 'varianceScaling',
            activation: this.data.activation as any
        }));

        this.data.model.setWeights(await this.decodeWeights(data.weights, data.specs));
            
        if (!minimal) {
            this.data.labels = data.labels;
            this.data.batchSize = data.batchSize;
            this.data.inputData = data.inputData;
            this.data.inputLabels = data.inputLabels;
            this.generateTestingData();
            this.compileModel();
        }
        return this;
    }

    // async import(json: string, minimal: boolean): Promise<Tensorflow1dCnnClassifier> {
    //     if (minimal) {
    //         const data = JSON.parse(json) as Tensorflow1dCnnClassifierMinimalData;
    //         this.data.labels = data.labels;
    //         const handler = new TensorflowIOHandler(data.model_config);
    //         this.data.model = await (tf.loadLayersModel(handler, { strict: true })) as tf.Sequential;
    //     } else {
    //         const data = JSON.parse(json) as Tensorflow1dCnnClassifierData;
    //         this.data.labels = data.labels;
    //         this.data.batchSize = data.batchSize;
    //         this.data.inputData = data.inputData;
    //         this.data.inputLabels = data.inputLabels;
    //         const handler = new TensorflowIOHandler(data.model_config);
    //         this.data.model = await (tf.loadLayersModel(handler, { strict: true })) as tf.Sequential;
    //         this.generateTestingData();
    //         this.compileModel();
    //     }
    //     return this;
    // }

    compileModel() {
        const optimizer = tf.train.adam();
        this.data.model.compile({
            optimizer: optimizer,
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy', 'mse'],
        });
    }

    test(argument: {
        [id: string]: any[];
    }): { [id: string]: any[] } {
        const argumentInput = (argument['input'] as number[][]);
        const tfData = tf.tensor2d(argumentInput, [1, argumentInput[0].length]);
        const testInput = tfData.reshape<tf.Rank.R3>([1, argumentInput[0].length, 1]);

        const testOutput = this.data.model.predict(testInput) as tf.Tensor<tf.Rank.R2>;
        const output = testOutput.arraySync()[0];
        return {
            'output': [this.data.labels[output.indexOf(Math.max(...output))]]
        }
    }

    generateTestingData() {
        const labels = this.data.inputLabels.map((value) => {
            const label = Array(this.data.labels.length).fill(0.0);
            label[this.data.labels.indexOf(value[0])] = 1.0;
            return label;
        });

        const tfData = tf.tensor2d(this.data.inputData, [this.data.inputData.length, this.data.inputData[0].length]);
        const tfLabels = tf.tensor2d(labels, [this.data.inputLabels.length, this.data.labels.length]);

        [this.data.trainX, this.data.trainLabels] = tf.tidy(() => {
            return [
                tfData.reshape<tf.Rank.R3>([this.data.inputData.length, this.data.inputData[0].length, 1]),
                tfLabels
            ];
        });
    }

    initialize() {
        this.generateTestingData();

        this.data.model = tf.sequential();

        this.data.model.add(tf.layers.conv1d({
            inputShape: [this.data.inputData[0].length, 1],
            kernelSize: 5,
            filters: 8,
            strides: 1,
            activation: 'relu',
            kernelInitializer: 'varianceScaling'
        }));
        this.data.model.add(tf.layers.maxPooling1d({ poolSize: 2, strides: 2 }));

        this.data.model.add(tf.layers.conv1d({
            kernelSize: 5,
            filters: 16,
            strides: 1,
            activation: 'relu',
            kernelInitializer: 'varianceScaling'
        }));
        this.data.model.add(tf.layers.maxPooling1d({ poolSize: 2, strides: 2 }));

        this.data.model.add(tf.layers.flatten());
        this.data.model.add(tf.layers.dense({
            units: this.data.labels.length,
            kernelInitializer: 'varianceScaling',
            activation: this.data.activation as any
        }));

        this.compileModel();
    }

    async step(): Promise<void> {
        (await this.data.model.fit(this.data.trainX, this.data.trainLabels, {
            batchSize: this.data.batchSize,
            epochs: 1,
            callbacks: {
                onBatchEnd: (batch, logs) => {
                    this.data.model.stopTraining = true;
                    if (logs) {
                        this.recorder?.record([
                            {
                                label: 'Mean Squared Error',
                                value: logs['mse'],
                            },
                            {
                                label: 'Accuracy',
                                value: logs['acc'],
                            },
                            {
                                label: 'Loss',
                                value: logs['loss'],
                            }
                        ]);
                    }
                }
            }
        }));
    }

    autoDetect(): number[] {
        const list = Array.from(new Set(this.data.inputLabels.map((value) => value[0])));
        list.sort((a, b) => a - b);
        return list;
    }

    setActivation(activation: string) {
        this.data.activation = activation;
    }

    setLabels(labels: number[]) {
        const list = Array.from(new Set(labels));
        list.sort((a, b) => a - b);
        this.data.labels = list;
    }

    setBatchSize(batch: number) {
        this.data.batchSize = batch;
    }

    getTestingInputs(): { input: PluginDataInput[], output?: PluginDataInput[] } {
        return {
            input: [
                {
                    id: 'input',
                    label: 'Testing Input Features',
                    min: 1,
                    max: 1,
                    type: 'number[]'
                }
            ],
            output: [
                {
                    id: 'output',
                    label: 'Testing Output Feature',
                    min: 1,
                    max: 1,
                    type: 'number'
                }
            ]
        }
    }
}

class Tensorflow1dCnnClassifierPluginOptions extends PluginOptions {

    state: number;
    labels: number[];

    constructor(public classifier: Tensorflow1dCnnClassifier) {
        super();
        this.state = 1;
        this.labels = [];
    }

    submit(inputs: { [id: string]: any; }): void {
        switch (this.state) {
            case 1:
                this.classifier.setActivation(inputs['activation']);
                this.classifier.setBatchSize(inputs['batch']);
                this.labels = this.classifier.autoDetect();
                this.state = 2;
                break;
            case 3:
                this.classifier.setLabels(JSON.parse(`[${inputs['labels']}]`));
                this.state = 4;
                break;
            default:
                throw new Error(`Tensorflow 1D CNN Classifion in invalid state.`);
        }
    }

    async executeCommand(id: string): Promise<void> {
        if (this.state === 2) {
            if (id === 'yes') {
                this.classifier.setLabels(this.labels);
                this.state = 4;
            } else if (id === 'no') {
                this.state = 3;
            } else {
                throw new Error(`Tensorflow 1D CNN Classification got invalid command: ${id}`);
            }
        } else {
            throw new Error(`Tensorflow 1D CNN Classifion in invalid state.`);
        }
    }

    options(): Option[] {
        switch (this.state) {
            case 1:
                return [
                    new ChoicesOption({
                        id: 'activation',
                        choices: [
                            'softmax',
                            'elu',
                            'hardSigmoid',
                            'relu',
                            'relu6',
                            'selu',
                            'sigmoid',
                            'softplus',
                            'softsign',
                            'tanh'
                        ],
                        label: 'Choose an activation function'
                    }),
                    new NumberOption({
                        id: 'batch',
                        label: 'Choose a batch size',
                        min: 1,
                        step: 100
                    })
                ];
            case 2:
                return [
                    new CommandOption({
                        id: 'yes',
                        command: 'Yes',
                        label: `Are these labels ${this.labels} correct?`,
                    }),
                    new CommandOption({
                        id: 'no',
                        command: 'No',
                        label: 'Incorrect. Will go to manual input when click',
                    })
                ];
            case 3:
                return [
                    new TextOption({
                        id: 'labels',
                        label: 'Input Label List. (example input: 1,2,3,4)',
                        min: 1,
                        pattern: '([ ]*[0-9]+[ ]*)(,[ ]*[0-9]+[ ]*)+'
                    })
                ];
            default:
                throw new Error(`Tensorflow 1D CNN Classifier in invalid state.`)
        }
    }

    noMore(): boolean {
        return this.state === 4;
    }

}

class Tensorflow1dCnnClassifierPluginInputs extends PluginInputs {
    constructor(public classifier: Tensorflow1dCnnClassifier) {
        super();
    }

    inputs(): PluginDataInput[] {
        return [
            {
                id: 'input',
                label: 'Input Array Feature',
                min: 1,
                max: 1,
                type: 'number[]'
            },
            {
                id: 'output',
                label: 'Output Feature',
                min: 1,
                max: 1,
                type: 'number'
            }
        ];
    }

    submit(inputs: { [id: string]: PluginData; }): void {
        if (inputs['input'] === undefined) {
            throw new Error(`Tensorflow 1D CNN Classifion's submit expecting plugin data with key input`);
        } else {
            this.classifier.setInput(inputs['input'].examples.map((value) => value[0] as number[]));
        }
        if (inputs['output'] === undefined) {
            throw new Error(`Tensorflow 1D CNN Classifion's submit expecting plugin data with key output`);
        } else {
            this.classifier.setOutput(inputs['output'].examples);
        }
    }
}