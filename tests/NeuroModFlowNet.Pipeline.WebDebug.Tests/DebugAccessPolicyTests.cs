using Microsoft.AspNetCore.Http;
using NeuroModFlowNet.Pipeline.WebDebug;

namespace NeuroModFlowNet.Pipeline.WebDebug.Tests;

public sealed class DebugAccessPolicyTests
{
    [Theory]
    [InlineData("http://127.0.0.1:443")]
    [InlineData("https://localhost")]
    [InlineData("http://0.0.0.0:443")]
    public void Port443IsAlwaysRejected(string urls) => Assert.Throws<InvalidOperationException>(() => new DebugAccessPolicy(urls,new string('x',32)));

    [Fact]
    public void RemoteBindRequiresTokenAndAllowsExplicitProjectPort()
    {
        Assert.Throws<InvalidOperationException>(() => new DebugAccessPolicy("http://0.0.0.0:5264",null));
        _ = new DebugAccessPolicy("http://0.0.0.0:5264",new string('x',32));
    }

    [Fact]
    public void TokenAndMutationHeaderAreCheckedIndependently()
    {
        string token = new('x',32);
        var policy = new DebugAccessPolicy("http://0.0.0.0:5264",token);
        var context = new DefaultHttpContext();
        context.Request.Method = "POST";
        Assert.Equal(401,policy.Reject(context.Request));
        context.Request.Headers.Authorization = "Bearer " + token;
        Assert.Equal(403,policy.Reject(context.Request));
        context.Request.Headers["X-VM-Debug"] = "1";
        Assert.Null(policy.Reject(context.Request));
        context.Request.Headers.Authorization = "Bearer " + new string('y',32);
        Assert.Equal(401,policy.Reject(context.Request));
    }
}
