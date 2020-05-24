module TestXgbc

using Test
using Random
using AutoMLPipeline
using AutoMLPipeline.XGBoostLearners
using AutoMLPipeline.CrossValidators
using AutoMLPipeline.Utils

acc(X,Y) = score(:accuracy,X,Y)
  
function test_xgbc()
  Random.seed!(123)
  data=getiris()
  X=data[:,1:4] 
  Y=data[:,5] |> collect
  xgb = Xgbc(Dict(:silent=>true))
  @test crossvalidate(xgb,X,Y,acc,10,false).mean > 90.0
end
@testset "Xgbc" begin
  test_xgbc()
end


end
