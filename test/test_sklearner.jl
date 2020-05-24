module TestSKL

using Random
using Test
using AutoMLPipeline.AbsTypes
using AutoMLPipeline.SKLearners
using AutoMLPipeline.SKPreprocessors
using AutoMLPipeline.Utils
using AutoMLPipeline.BaseFilters
using AutoMLPipeline.Pipelines
using AutoMLPipeline.EnsembleMethods
using AutoMLPipeline.FeatureSelectors
using Statistics
using DataFrames

const IRIS = getiris()
const X = IRIS[:,1:3] |> DataFrame
const XC = IRIS[:,1:4] |> DataFrame
const YC = IRIS[:,5] |> Vector
const YN = IRIS[:,4] |> Vector

function fit_test(learner::String,in::DataFrame,out::Vector)
  _learner=SKLearner(learner)
  fit!(_learner,in,out)
  @test _learner.model != Dict()
  return _learner
end

function fit_transform_reg(model::Learner,in::DataFrame,out::Vector)
  @test sum((transform!(model,in) .- out).^2)/length(out) < 2.0
end


const cclassifiers = [
		"LinearSVC","QDA","MLPClassifier","BernoulliNB",
		"RandomForestClassifier",		
		"NearestCentroid","SVC","LinearSVC","NuSVC","MLPClassifier",
		"SGDClassifier","KNeighborsClassifier",
		"DecisionTreeClassifier",
		"PassiveAggressiveClassifier","RidgeClassifier",
		"ExtraTreesClassifier","GradientBoostingClassifier",
		"BaggingClassifier","AdaBoostClassifier","GaussianNB","MultinomialNB",
		"ComplementNB","BernoulliNB"
		#"RidgeClassifierCV",
		#"GaussianProcessClassifier",
		#"LDA",
	  ]
@testset "scikit classifiers" begin
  Random.seed!(123)
  for cl in cclassifiers
	 #println(cl)
	 fit_test(cl,XC,YC)
  end
end

const cregressors = [
	  "SVR", "Ridge", "Lasso", "ElasticNet", "Lars", "LassoLars",
	  "OrthogonalMatchingPursuit",
	  "SGDRegressor", "PassiveAggressiveRegressor",
	  "KNeighborsRegressor", "RadiusNeighborsRegressor",
	  "DecisionTreeRegressor", "RandomForestRegressor", 
	  "ExtraTreesRegressor", "GradientBoostingRegressor",
	  "AdaBoostRegressor"
	  #"GaussianProcessRegressor",
	  #"KernelRidge",
	  #"BayesianRidge",
	  #"ARDRegression",
	  #"RidgeCV",
	  #"MLPRegressor",
	 ]
@testset "scikit regressors" begin
  Random.seed!(123)
  for rg in cregressors
	 #println(rg)
	 model=fit_test(rg,X,YN)
	 fit_transform_reg(model,X,YN)
  end
end

end
