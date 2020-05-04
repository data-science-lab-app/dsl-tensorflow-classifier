import * as tf from '@tensorflow/tfjs';

interface ModelJson {
    artifact: tf.io.ModelArtifacts,
    weights?: string;
}

export class TensorflowIOHandler implements tf.io.IOHandler {
    constructor(public json?: string) {

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
        const json: ModelJson = {
            artifact,
            weights: await this.encodeWeights(model.getWeights())
        };
        this.json = JSON.stringify(json);
        return this.json;
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
        })
        return Buffer.from(y.buffer).toString('binary');
    }

    get save(): tf.io.SaveHandler {
        return (modelArtifact: tf.io.ModelArtifacts): Promise<tf.io.SaveResult> => {
            return new Promise<tf.io.SaveResult>(async (resolve) => {
                const data: ModelJson = {
                    artifact: modelArtifact,
                };
                const weightData = modelArtifact.weightData;
                if (weightData) {
                    data.weights = await this.saveWeights(weightData);
                    data.artifact.weightData = undefined;
                }
                this.json = JSON.stringify(data);
                resolve();
            });
        }
    };
    get load(): tf.io.LoadHandler {
        return (): Promise<tf.io.ModelArtifacts> => {
            return new Promise<tf.io.ModelArtifacts>(async (resolve, reject) => {
                if (this.json) {
                    const modelJson = JSON.parse(this.json) as ModelJson;
                    const modelArtifacts: tf.io.ModelArtifacts = {
                        modelTopology: modelJson.artifact.modelTopology,
                        format: modelJson.artifact.format,
                        generatedBy: modelJson.artifact.generatedBy,
                        convertedBy: modelJson.artifact.convertedBy,
                        trainingConfig: modelJson.artifact.trainingConfig,
                        userDefinedMetadata: modelJson.artifact.userDefinedMetadata
                    }
                    if (modelJson.weights !== undefined) {
                        modelArtifacts.weightData = await this.loadWeights(modelJson.weights);
                        modelArtifacts.weightSpecs = modelJson.artifact.weightSpecs;
                    }
                    resolve(modelArtifacts);
                } else {
                    reject(new Error(`Load requires a json object!`));
                }
            });
        }
    }

    async saveWeights(weights: ArrayBuffer): Promise<string> {
        return Buffer.from(weights).toString('binary');
    }

    async loadWeights(weights: string): Promise<ArrayBuffer> {
        const buffer = Buffer.from(weights, 'binary');
        return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
    }
}