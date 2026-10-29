using System.Security.Cryptography;
using System.Text;

namespace NeuroModFlowNet.Pipeline.WebDebug;

internal sealed class DebugAccessPolicy
{
    readonly byte[]? token;

    public DebugAccessPolicy(string urls, string? accessToken)
    {
        if(accessToken is { Length: < 24 }) throw new InvalidOperationException("Access token must contain at least 24 characters.");
        foreach(string address in urls.Split(';', StringSplitOptions.RemoveEmptyEntries))
        {
            if(!Uri.TryCreate(address.Replace("*", "0.0.0.0").Replace("+", "0.0.0.0"), UriKind.Absolute, out Uri? uri) ||
                uri.Scheme is not ("http" or "https") || uri.Port < 1024)
                throw new InvalidOperationException("Use an explicit HTTP(S) project address and port other than 443.");
            if(!uri.IsLoopback && string.IsNullOrWhiteSpace(accessToken))
                throw new InvalidOperationException("Remote binding requires WebDebug:AccessToken (at least 24 characters).");
        }
        token = accessToken is null ? null : Encoding.UTF8.GetBytes(accessToken);
    }

    public int? Reject(HttpRequest request)
    {
        if(token is not null)
        {
            string authorization = request.Headers.Authorization.ToString();
            if(!authorization.StartsWith("Bearer ", StringComparison.Ordinal) ||
                !CryptographicOperations.FixedTimeEquals(Encoding.UTF8.GetBytes(authorization[7..]), token))
                return StatusCodes.Status401Unauthorized;
        }
        return request.Method != "GET" && request.Headers["X-VM-Debug"] != "1"
            ? StatusCodes.Status403Forbidden : null;
    }
}
