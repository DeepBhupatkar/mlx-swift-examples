//
//  Gemma3.swift
//  mlx-swift-examples
//
//  Created by Deep Bhupatkar on 26/3/25.
//

// port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/gemma3

import CoreImage
import Foundation
import Hub
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import Tokenizers

// MARK: - Language

private enum Language {
    // specialized norm for gemma
    fileprivate class RMSNorm: Module, UnaryLayer {
        let weight: MLXArray
        let eps: Float

        public init(dimensions: Int, eps: Float = 1e-5) {
            self.weight = MLXArray.ones([dimensions]).asType(.float16)
            self.eps = eps
            super.init()
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            return MLXFast.rmsNorm(x, weight: 1.0 + self.weight, eps: self.eps)
        }
    }

    fileprivate class Attention: Module {
        let args: Gemma3Configuration.TextConfiguration
        let scale: Float

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        let rope: RoPE

        public init(_ args: Gemma3Configuration.TextConfiguration) {
            self.args = args

            let dim = args.hiddenSize
            let heads = args.attentionHeads
            let kvHeads = args.kvHeads

            let headDim = args.hiddenSize / heads
            self.scale = pow(Float(headDim), -0.5)

            self._wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
            self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            self._wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

            self.rope = RoPE(
                dimensions: headDim, traditional: args.ropeTraditional, base: args.ropeTheta)
        }

        public func callAsFunction(
            _ x: MLXArray, mask: MLXArray? = nil, cache: KVCache?
        ) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))

            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            // prepare the queries, keys and values for the attention computation
            queries = queries.reshaped(B, L, args.attentionHeads, -1).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

            if let cache {
                queries = rope(queries, offset: cache.offset)
                keys = rope(keys, offset: cache.offset)
                (keys, values) = cache.update(keys: keys, values: values)
            } else {
                queries = rope(queries)
                keys = rope(keys)
            }

            let output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return wo(output)
        }
    }

    fileprivate class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "down_proj") var down: Linear
        @ModuleInfo(key: "up_proj") var up: Linear

        public init(dimensions: Int, hiddenDimensions: Int) {
            self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
            self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(gelu(gate(x)) * up(x))
        }
    }

    fileprivate class TransformerBlock: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        let mlp: MLP

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        public init(_ args: Gemma3Configuration.TextConfiguration) {
            self._attention.wrappedValue = Attention(args)
            self.mlp = MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
            self._inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            self._postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
        }

        public func callAsFunction(
            _ x: MLXArray, mask: MLXArray? = nil, cache: KVCache?
        ) -> MLXArray {
            var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
            let h = x + r
            r = mlp(postAttentionLayerNorm(h))
            let out = h + r
            return out
        }
    }
    
    fileprivate class GemmaModel: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

        fileprivate let layers: [TransformerBlock]
        fileprivate let norm: RMSNorm

        let hiddenScale: Float

        public init(_ args: Gemma3Configuration.TextConfiguration) {
            precondition(args.vocabularySize > 0)

            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

            self.hiddenScale = pow(Float(args.hiddenSize), 0.5)

            self.layers = (0 ..< args.hiddenLayers)
                .map { _ in
                    TransformerBlock(args)
                }
            self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        }

        public func callAsFunction(
            _ inputs: MLXArray, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil,
            mask: MLXArray? = nil
        ) -> MLXArray {
            var h = inputEmbedding ?? embedTokens(inputs)
            h = h * hiddenScale

            let mask: MLXArray? =
                if mask == nil || (cache?[0].offset ?? 0) > 0 {
                    createAttentionMask(h: h, cache: cache)
                } else {
                    nil
                }

            for (i, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[i])
            }

            return norm(h)
        }
    }

    fileprivate class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo var model: GemmaModel

        var kvHeads: [Int]
        var headDim: MLX.IntOrPair

        public init(_ args: Gemma3Configuration.TextConfiguration) {
            self.model = GemmaModel(args)

            self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
            self.headDim = IntOrPair(args.hiddenSize / args.attentionHeads)
        }

        public func callAsFunction(
            _ inputs: MLXArray, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil,
            mask: MLXArray? = nil
        ) -> LMOutput {
            var out = model(inputs, cache: cache, inputEmbedding: inputEmbedding, mask: mask)
            out = model.embedTokens.asLinear(out)
            return LMOutput(logits: out)
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            weights.filter {
                !$0.key.contains("self_attn.rotary_emb.inv_freq")
            }
        }
    }
}

// MARK: - Vision

private enum Vision {
    fileprivate class Attention: Module {
        let numHeads: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "out_proj") var wo: Linear

        public init(dims: Int, numHeads: Int, bias: Bool = true) {
            precondition(dims % numHeads == 0, "Dimensions must be divisible by numHeads")

            self.numHeads = numHeads
            let headDim = dims / numHeads
            self.scale = pow(Float(headDim), -0.5)

            self._wq.wrappedValue = Linear(dims, dims, bias: bias)
            self._wk.wrappedValue = Linear(dims, dims, bias: bias)
            self._wv.wrappedValue = Linear(dims, dims, bias: bias)
            self._wo.wrappedValue = Linear(dims, dims, bias: bias)
        }

        public func callAsFunction(
            _ x: MLXArray, mask: MLXArray? = nil
        ) -> MLXArray {
            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            let (B, L) = (queries.dim(0), queries.dim(1))
            let D = queries.dim(2) / numHeads

            queries = queries.reshaped(B, L, numHeads, D).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, L, numHeads, D).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, numHeads, D).transposed(0, 2, 1, 3)

            let output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return wo(output)
        }
    }

    fileprivate class LayerNorm: Module, UnaryLayer {
        @ModuleInfo(key: "weight") var weight: MLXArray
        @ModuleInfo(key: "bias") var bias: MLXArray

        public init(dimensions: Int, eps: Float = 1e-5) {
            self._weight.wrappedValue = ones([dimensions])
            self._bias.wrappedValue = zeros([dimensions])
            super.init()
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            layerNorm(x, weight: weight, bias: bias, eps: 1e-5)
        }
    }

    fileprivate class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "fc1") var fc1: Linear
        @ModuleInfo(key: "fc2") var fc2: Linear
        @ModuleInfo(key: "act_fn") var actFn: GELU

        public init(inputDim: Int, hiddenDim: Int) {
            self._fc1.wrappedValue = Linear(inputDim, hiddenDim)
            self._fc2.wrappedValue = Linear(hiddenDim, inputDim)
            self._actFn.wrappedValue = GELU()
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(actFn(fc1(x)))
        }
    }

    fileprivate class VisionEncoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
        @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm
        let mlp: MLP

        public init(config: Gemma3Configuration.VisionConfiguration) {
            self._attention.wrappedValue = Attention(
                dims: config.hiddenSize, numHeads: config.attentionHeads)
            self._layerNorm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
            self._layerNorm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
            self.mlp = MLP(inputDim: config.hiddenSize, hiddenDim: config.intermediateSize)
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            var out = x
            let residual = out
            out = layerNorm1(out)
            out = attention(out)
            out = out + residual

            let residual2 = out
            out = layerNorm2(out)
            out = mlp(out)
            out = out + residual2

            return out
        }
    }

    fileprivate class InternalVisionModel: Module {
        @ModuleInfo(key: "embeddings") var embeddings: Patch2EmbeddingWithPositionEmbeddings
        @ModuleInfo(key: "encoder") var encoder: VisionEncoder
        @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm

        public init(_ config: Gemma3Configuration.VisionConfiguration) {
            self._embeddings.wrappedValue = Patch2EmbeddingWithPositionEmbeddings(config)
            self._encoder.wrappedValue = VisionEncoder(config)
            self._postLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        }

        public func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool = false) -> (
            MLXArray, MLXArray, MLXArray?
        ) {
            let embeds = embeddings(x)
            let (pooled, encoderOutput) = encoder(embeds, outputHiddenStates: outputHiddenStates)
            let pooledOutput = postLayerNorm(pooled)

            return (pooledOutput, embeds, encoderOutput)
        }
    }

    fileprivate class Patch2EmbeddingWithPositionEmbeddings: Module {
        @ModuleInfo(key: "patch_embeddings") var patchEmbeddings: PatchEmbeddings
        @ModuleInfo(key: "position_embeddings") var positionEmbeddings: MLXArray
        @ModuleInfo(key: "pre_layernorm") var preLayerNorm: LayerNorm

        public init(_ config: Gemma3Configuration.VisionConfiguration) {
            self._patchEmbeddings.wrappedValue = PatchEmbeddings(config)
            self._preLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize)

            var numPatches = (config.imageSize / config.patchSize) * (config.imageSize / config.patchSize)
            if config.useCLS {
                numPatches += 1
            }
            self._positionEmbeddings.wrappedValue = zeros([1, numPatches, config.hiddenSize])
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            var embeddings = patchEmbeddings(x)
            embeddings = embeddings + positionEmbeddings
            embeddings = preLayerNorm(embeddings)
            return embeddings
        }
    }

    fileprivate class PatchEmbeddings: Module {
        @ModuleInfo(key: "projection") var projection: Conv2d
        let config: Gemma3Configuration.VisionConfiguration
        
        public init(_ config: Gemma3Configuration.VisionConfiguration) {
            self.config = config
            
            self._projection.wrappedValue = Conv2d(
                inputChannels: 3,
                outputChannels: config.hiddenSize,
                kernelSize: IntOrPair(config.patchSize),
                stride: IntOrPair(config.patchSize),
                padding: IntOrPair(0),
                bias: true
            )
        }
        
        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (batchSize, _, _, _) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
            var out = projection(x)
            
            out = out.reshaped(batchSize, config.hiddenSize, -1)
            out = out.transposed(0, 2, 1)
            
            if config.useCLS {
                // Prepend CLS token
                let clsTokens = zeros([batchSize, 1, config.hiddenSize])
                out = MLX.concatenated([clsTokens, out], axis: 1)
            }
            
            return out
        }
    }

    fileprivate class VisionEncoder: Module {
        @ModuleInfo(key: "layers") var layers: [VisionEncoderLayer]
        let config: Gemma3Configuration.VisionConfiguration

        public init(_ config: Gemma3Configuration.VisionConfiguration) {
            self.config = config
            
            self._layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
                VisionEncoderLayer(config: config)
            }
        }

        public func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool = false) -> (MLXArray, MLXArray?) {
            var hiddenStates: MLXArray? = nil
            var out = x

            for layer in layers {
                out = layer(out)
            }

            if outputHiddenStates {
                hiddenStates = out
            }

            // Use the CLS token if available, otherwise average pool
            let pooledOutput: MLXArray
            if config.useCLS {
                pooledOutput = out[0..., 0, 0...]
            } else {
                pooledOutput = mean(out, axis: 1)
            }

            return (pooledOutput, hiddenStates)
        }
    }

    fileprivate class VisionModel: Module {
        @ModuleInfo(key: "vision_model") var visionModel: InternalVisionModel

        public init(_ config: Gemma3Configuration.VisionConfiguration) {
            self._visionModel.wrappedValue = InternalVisionModel(config)
        }

        public func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool = false) -> (
            MLXArray, MLXArray, MLXArray?
        ) {
            visionModel(x, outputHiddenStates: outputHiddenStates)
        }
    }
}

// MARK: - Gemma3 Configuration

public struct Gemma3Configuration: Codable, Sendable {
    public struct TextConfiguration: Codable, Sendable {
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let ropeTheta: Float
        public let ropeTraditional: Bool
        public let vocabularySize: Int
        
        // Some values might need defaults
        public let _rmsNormEps: Float?
        public var rmsNormEps: Float { _rmsNormEps ?? 1e-6 }

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case hiddenLayers = "num_hidden_layers"
            // Try multiple possible key names for attention heads
            case attentionHeads = "num_attention_heads" 
            case attentionHeadsAlt1 = "n_heads"
            case attentionHeadsAlt2 = "attention_heads"
            case kvHeads = "num_key_value_heads"
            case kvHeadsAlt = "kv_heads"
            case ropeTheta = "rope_theta"
            case ropeTraditional = "rope_traditional"
            case vocabularySize = "vocab_size"
            case _rmsNormEps = "rms_norm_eps"
        }
        
        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            
            hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
            
            // Handle optional fields with defaults
            if let intermediateSizeValue = try? container.decode(Int.self, forKey: .intermediateSize) {
                intermediateSize = intermediateSizeValue
            } else {
                // Common default: 4x the hidden size
                intermediateSize = hiddenSize * 4
            }
            
            hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
            
            // Try different keys for attention heads
            if let heads = try? container.decode(Int.self, forKey: .attentionHeads) {
                attentionHeads = heads
            } else if let heads = try? container.decode(Int.self, forKey: .attentionHeadsAlt1) {
                attentionHeads = heads
            } else if let heads = try? container.decode(Int.self, forKey: .attentionHeadsAlt2) {
                attentionHeads = heads
            } else {
                // Default to a reasonable value based on hidden size
                attentionHeads = hiddenSize / 64
            }
            
            // Try different keys for KV heads
            if let heads = try? container.decode(Int.self, forKey: .kvHeads) {
                kvHeads = heads
            } else if let heads = try? container.decode(Int.self, forKey: .kvHeadsAlt) {
                kvHeads = heads
            } else {
                // Default to same as attention heads
                kvHeads = attentionHeads
            }
            
            // Try to decode or provide reasonable defaults
            ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
            ropeTraditional = try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
            vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
            _rmsNormEps = try container.decodeIfPresent(Float.self, forKey: ._rmsNormEps)
        }
    }
    
    public struct VisionConfiguration: Codable, Sendable {
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let numHiddenLayers: Int
        public let attentionHeads: Int
        public let imageSize: Int
        public let patchSize: Int
        public let useCLS: Bool
        
        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case numHiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case attentionHeadsAlt1 = "n_heads"
            case attentionHeadsAlt2 = "attention_heads"
            case imageSize = "image_size"
            case patchSize = "patch_size"
            case useCLS = "use_cls_token"
        }
        
        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            
            hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
            
            // Handle optional fields with defaults
            if let intermediateSizeValue = try? container.decode(Int.self, forKey: .intermediateSize) {
                intermediateSize = intermediateSizeValue
            } else {
                // Common default: 4x the hidden size
                intermediateSize = hiddenSize * 4
            }
            
            numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
            
            // Try different keys for attention heads
            if let heads = try? container.decode(Int.self, forKey: .attentionHeads) {
                attentionHeads = heads
            } else if let heads = try? container.decode(Int.self, forKey: .attentionHeadsAlt1) {
                attentionHeads = heads
            } else if let heads = try? container.decode(Int.self, forKey: .attentionHeadsAlt2) {
                attentionHeads = heads
            } else {
                // Default to a reasonable value based on hidden size
                attentionHeads = hiddenSize / 64
            }
            
            imageSize = try container.decode(Int.self, forKey: .imageSize)
            patchSize = try container.decode(Int.self, forKey: .patchSize)
            useCLS = try container.decodeIfPresent(Bool.self, forKey: .useCLS) ?? true
        }
    }
    
    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let vocabularySize: Int
    
    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case vocabularySize = "vocab_size"
    }
    
    // Make model-level vocabulary size optional, falling back to text config value
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        
        textConfiguration = try container.decode(TextConfiguration.self, forKey: .textConfiguration)
        visionConfiguration = try container.decode(VisionConfiguration.self, forKey: .visionConfiguration)
        
        // If vocab_size at the top level doesn't exist, fall back to the text config's value
        if let vocabSize = try? container.decode(Int.self, forKey: .vocabularySize) {
            vocabularySize = vocabSize
        } else {
            vocabularySize = textConfiguration.vocabularySize
        }
    }
}

// MARK: - Gemma3 Model

public class Gemma3: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionModel: Vision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Language.LanguageModel
    
    public let config: Gemma3Configuration
    
    public var vocabularySize: Int { config.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }
    public var headDim: MLX.IntOrPair { languageModel.headDim }
    
    public func loraLinearLayers() -> MLXLMCommon.LoRALinearLayers {
        languageModel.model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }
    
    public init(_ config: Gemma3Configuration) {
        self.config = config
        self._visionModel.wrappedValue = Vision.VisionModel(config.visionConfiguration)
        self._languageModel.wrappedValue = Language.LanguageModel(config.textConfiguration)
    }
    
    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws -> PrepareResult {
        guard let image = input.image else { throw VLMError.imageRequired }
        guard let mask = input.text.mask else { throw VLMError.maskRequired }
        let inputIds = input.text.tokens
        
        let inputEmbedding = inputEmbeddings(
            inputIds: inputIds, pixelValues: image.pixels, mask: mask)
        
        let result = languageModel(
            inputIds, cache: cache, inputEmbedding: inputEmbedding, mask: mask)
        
        return .logits(result)
    }
    
    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache).logits
    }
    
    private func inputEmbeddings(
        inputIds: MLXArray, pixelValues: MLXArray, mask: MLXArray
    ) -> MLXArray {
        let (_, imageEmbeddings, _) = visionModel(pixelValues)
        
        // Create token embeddings
        let tokenEmbeddings = languageModel.model.embedTokens(inputIds)
        
        // Replace image tokens with vision embeddings
        var hiddenStates = tokenEmbeddings
        
        // Find the image tokens in the input
        // Based on the mask and pattern, we know that the image tokens are at the beginning
        // This simplification assumes the input pattern is <image>...<image><bos>text
        let numImageTokens = imageEmbeddings.dim(1)
        
        // Replace the token embeddings with the image embeddings for the image tokens
        if numImageTokens > 0 {
            // The first tokens should be replaced with image embeddings
            let batchSize = hiddenStates.dim(0)
            for b in 0..<batchSize {
                let imgEmbed = imageEmbeddings[b..<b+1]
                hiddenStates[b..<b+1, 0..<numImageTokens] = imgEmbed
            }
        }
        
        return hiddenStates
    }
}

// MARK: - Gemma3 Processor

public struct Gemma3ProcessorConfiguration: Codable, Sendable {
    public struct Size: Codable, Sendable {
        public let width: Int
        public let height: Int
        
        var cgSize: CGSize { .init(width: width, height: height) }
    }
    
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let size: Size
    public let imageSequenceLength: Int
    
    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }
    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }
    
    enum CodingKeys: String, CodingKey {
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case size
        case imageSequenceLength = "image_sequence_length"
    }
}

public class Gemma3Processor: UserInputProcessor {
    private let config: Gemma3ProcessorConfiguration
    private let tokenizer: any Tokenizer
    
    public init(_ config: Gemma3ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }
    
    private func prepare(image: CIImage, processing: UserInput.Processing?) -> MLXArray {
        // Based on image_processing from transformers
        var image = image
        
        // We want to do all of the image processing in an sRGB tone curve
        // rather than a linear space as that is what transformers / torch_vision
        // do (implicitly by using sRGB rasters directly)
        image = MediaProcessing.inSRGBToneCurveSpace(image)
        
        // Apply user instructions
        image = MediaProcessing.apply(image, processing: processing)
        
        image = MediaProcessing.resampleBicubic(image, to: config.size.cgSize)
        image = MediaProcessing.normalize(
            image, mean: config.imageMeanTuple, std: config.imageStdTuple)
        
        return MediaProcessing.asMLXArray(image)
    }
    
    public func prepare(input: UserInput) throws -> LMInput {
        switch input.images.count {
        case 0: throw VLMError.imageRequired
        case 1: break
        default: throw VLMError.singleImageAllowed
        }
        
        // For Gemma3, get the last message content
        var prompt = input.prompt.asMessages().last?["content"] as? String ?? ""
        
        // Insert image tokens at the beginning of the prompt
        let count = input.images.count * config.imageSequenceLength
        let bosToken = tokenizer.bosToken ?? ""
        prompt = Array(repeating: "<image>", count: count).joined() + bosToken + prompt
        
        let promptTokens = try tokenizer.encode(text: prompt)
        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray)
        
        let pixels = try prepare(image: input.images[0].asCIImage(), processing: input.processing)
        
        return LMInput(text: .init(tokens: promptArray, mask: mask), image: .init(pixels: pixels))
    }
} 
