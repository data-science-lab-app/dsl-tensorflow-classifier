import * as tf from '@tensorflow/tfjs';

interface ModelJson {
    artifact: tf.io.ModelArtifacts,
    weights?: string
}

export class TensorflowIOHandler implements tf.io.IOHandler {
    constructor(public json?: string) {

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
                resolve({
                    modelArtifactsInfo: {
                        dateSaved: new Date(),
                        modelTopologyType: 'JSON',
                        modelTopologyBytes: modelArtifact.modelTopology == null ? 0 : Buffer.byteLength(JSON.stringify(modelArtifact.modelTopology), 'utf-8'),
                        weightSpecsBytes: modelArtifact.weightSpecs == null ? 0 : Buffer.byteLength(JSON.stringify(modelArtifact.weightSpecs), 'utf-8'),
                        weightDataBytes: weightData == null ? 0 : weightData.byteLength
                    }
                });
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
        const buffer = Buffer.from(weights);
        return JSON.stringify(buffer.toJSON());
    }

    async loadWeights(weights: string): Promise<ArrayBuffer> {
        var buffer = Buffer.from(JSON.parse(weights));
        const arrayBuffer = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
        return arrayBuffer;
    }
}