import { AlgorithmPlugin, PluginOptions, PluginInputs, Option, CheckboxOption, RecorderService, PluginData, PluginDataInput, NumberOption, ChoicesOption, CommandOption, TextOption } from 'data-science-lab-core';
import * as tf from '@tensorflow/tfjs';

import * as tf_node from '@tensorflow/tfjs-node';
import { nextFrame } from '@tensorflow/tfjs';

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

    async export(minimal: boolean): Promise<string> {
        return '';
    }

    async import(json: string, minimal: boolean): Promise<Tensorflow1dCnnClassifier> {
        return this;
    }

    test(argument: {
        [id: string]: any[];
    }): { [id: string]: any[] } {
        throw new Error('Not implemented');
    }

    initialize() {
        const labels = this.data.inputLabels.map((value) => {
            const label = Array(10).fill(0.0);
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

        const optimizer = tf.train.adam();
        this.data.model.compile({
            optimizer: optimizer,
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy'],
        });
    }

    async step(): Promise<void> {
        (await this.data.model.fit(this.data.trainX, this.data.trainLabels, {
            batchSize: this.data.batchSize,
            epochs: 1,
            yieldEvery: 125,
            callbacks: {
                onBatchEnd: () => {
                    this.data.model.stopTraining = true;
                },
                onYield: async (batch, logs) => {
                    await tf.nextFrame()
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
                        min: 100,
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