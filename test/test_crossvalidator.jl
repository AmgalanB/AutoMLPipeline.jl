module TestCrossValidator

using Test
using Random
using AutoMLPipeline
using AutoMLPipeline.CrossValidators
using AutoMLPipeline.DecisionTreeLearners
using AutoMLPipeline.Pipelines
using AutoMLPipeline.BaseFilters
using AutoMLPipeline.SKPreprocessors
using AutoMLPipeline.JLPreprocessors
using AutoMLPipeline.Utils

function test_crossvalidator()
  racc = 50.0
  Random.seed!(123)
  acc(X,Y) = score(:accuracy,X,Y)
  data=getiris()
  X=data[:,1:4] 
  Y=data[:,5] |> collect

  rf = RandomForest()
  rb  = SKPreprocessor("RobustScaler")
  stdsc= SKPreprocessor("StandardScaler")
  pca  = JLPreprocessor("PCA")
  ppca = JLPreprocessor("PPCA")
  kpca = JLPreprocessor("KernelPCA")
  fa  = JLPreprocessor("FA")
  ica = JLPreprocessor("ICA")
  ohe = OneHotEncoder()

  @test crossvalidate(rf,X,Y,acc,10,false).mean > racc

  Random.seed!(123)
  ppl1 = Pipeline([RandomForest()])
  @test crossvalidate(ppl1,X,Y,acc,10,false).mean > racc

  Random.seed!(123)
  ppl2 = @pipeline ohe |> stdsc |> rf
  @test crossvalidate(ppl2,X,Y,acc,10,false).mean > racc

  Random.seed!(123)
  ppl3 = @pipeline rb |> pca |> rf
  @test crossvalidate(ppl3,X,Y,acc,10,false).mean > racc

  Random.seed!(123)
  ppl5 = @pipeline stdsc |> ica |> rf
  @test crossvalidate(ppl5,X,Y,acc,10,false).mean > racc

  ica = JLPreprocessor("ICA",Dict(:autocomponent=>true))
  ppl6 = @pipeline rb |> ica |> rf
  @test crossvalidate(ppl6,X,Y,acc,10,false).mean == -Inf

  ppca = JLPreprocessor("PPCA",Dict(:autocomponent=>true))
  ppl7 = @pipeline rb |> ppca |> rf
  @test crossvalidate(ppl7,X,Y,acc,10,false).mean > racc

  kpca = JLPreprocessor("KernelPCA",Dict(:autocomponent=>true))
  ppl8 = @pipeline stdsc |> kpca |> rf
  @test crossvalidate(ppl8,X,Y,acc,10,false).mean > racc

  fa = JLPreprocessor("FA",Dict(:autocomponent=>true))
  ppl9 = @pipeline stdsc |> fa |> rf
  @test crossvalidate(ppl9,X,Y,acc,10,false).mean > racc

end
@testset "CrossValidator" begin
  test_crossvalidator()
end


end
