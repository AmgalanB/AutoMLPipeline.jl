module TestSKPreprocessing

using Random
using Test
using AutoMLPipeline
using AutoMLPipeline.SKPreprocessors
using AutoMLPipeline.JLPreprocessors
using AutoMLPipeline.Utils
using AutoMLPipeline.BaseFilters
using AutoMLPipeline.Pipelines
using AutoMLPipeline.EnsembleMethods
using AutoMLPipeline.FeatureSelectors
using Statistics
using DataFrames

Random.seed!(1)

const IRIS = getiris()
extra = rand(150,3) |> DataFrame
const X = hcat(IRIS[:,1:4],extra) |> DataFrame
const Y = IRIS[:,5] |> Vector

# "KernelCenterer","MissingIndicator","KBinsDiscretizer","OneHotEncoder", 

function fit_test(preproc::String,in::DataFrame,out::Vector)
  _preproc=SKPreprocessor(Dict(:preprocessor=>preproc))
  fit!(_preproc,in,out)
  @test _preproc.model != Dict()
  return _preproc
end

function transform_test(preproc::String,in::DataFrame,out::Vector)
  _preproc=SKPreprocessor(Dict(:preprocessor=>preproc))
  fit!(_preproc,in,out)
  res = transform!(_preproc,in)
  @test size(res)[1] == size(out)[1]
end

cpreprocessors = [
	  "FastICA", "IncrementalPCA", "LatentDirichletAllocation", "VarianceThreshold", "SimpleImputer",
	  "Binarizer", "FunctionTransformer", "MultiLabelBinarizer", "MaxAbsScaler",
	  "MinMaxScaler", "Normalizer", "OrdinalEncoder", "PolynomialFeatures",
	  "PowerTransformer", "QuantileTransformer", "RobustScaler", "StandardScaler"
	  #"MiniBatchDictionaryLearning",
	  #"MiniBatchSparsePCA",
	  #"NMF",
	  #"PCA",
	  #"KernelPCA",
	  #"TruncatedSVD",
	  #"DictionaryLearning",
	  #"FactorAnalysis",
	  ]
@testset "scikit preprocessors fit test" begin
  Random.seed!(123)
  for cl in cpreprocessors
	 #println(cl)
	 fit_test(cl,X,Y)
  end
end

@testset "scikit preprocessors transform test" begin
  Random.seed!(123)
  for cl in cpreprocessors
	 #println(cl)
	 transform_test(cl,X,Y)
  end
end

function skptest()
  features = X
  labels = Y

  pca = JLPreprocessor(Dict(:preprocessor=>"PCA"))
  fit!(pca,features)
  @test transform!(pca,features) |> x->size(x,2) == 3

  pca = JLPreprocessor("PCA",Dict(:autocomponent=>true))
  fit!(pca,features)
  @test transform!(pca,features) |> x->size(x,2) == 3

  pca = JLPreprocessor("PCA",Dict((:autocomponent=>false,:n_components=>2)))
  fit!(pca,features)
  @test transform!(pca,features) |> x->size(x,2) == 2 

  ica = JLPreprocessor("ICA",Dict(:autocomponent=>false,:n_components=>2))
  fit!(ica,features)
  @test transform!(ica,features) |> x->size(x,2) == 2

  stdsc = SKPreprocessor("StandardScaler")
  fit!(stdsc,features)
  @test abs(mean(transform!(stdsc,features) |> Matrix)) < 0.00001

  minmax = SKPreprocessor("MinMaxScaler")
  fit!(minmax,features)
  @test mean(transform!(minmax,features) |> Matrix) > 0.30

  vote = VoteEnsemble()
  stack = StackEnsemble()
  best = BestLearner()
  cat = CatFeatureSelector()
  num = NumFeatureSelector()
  disc = CatNumDiscriminator()
  ohe = OneHotEncoder()

  mpipeline =@pipeline stdsc |> pca |> best
  pred = fit_transform!(mpipeline,features,labels)
  @test score(:accuracy,pred,labels) > 50.0

  fpipe = @pipeline ((cat + num) + (num + pca))  |> stack
  fit!(fpipe,features,labels)
  @test ((transform!(fpipe,features) .== labels) |> sum ) / nrow(features) > 0.50

end
@testset "scikit preprocessor fit/transform test with real data" begin
  Random.seed!(123)
  skptest()
end


end
