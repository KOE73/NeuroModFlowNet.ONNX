using System.Text.Json.Serialization;
using NeuroModFlowNet.Pipeline.WebDebug;

if(args is ["--measure-register-catalog", string outputPath])
{
    RegisterCatalogBenchmark.Run(outputPath);
    return;
}

var builder = WebApplication.CreateBuilder(args);
if(builder.Configuration.GetSection("Kestrel:Endpoints").GetChildren().Any())
    throw new InvalidOperationException("Configure this debug host through --urls only, so port and remote access rules are validated.");
string urls = builder.Configuration["urls"] ?? "http://127.0.0.1:5264";
string? accessToken = builder.Configuration["WebDebug:AccessToken"];
var access = new DebugAccessPolicy(urls, accessToken);
builder.WebHost.UseUrls(urls);
builder.Services.ConfigureHttpJsonOptions(options => options.SerializerOptions.Converters.Add(new JsonStringEnumConverter()));
builder.Services.AddSingleton<DebugRuntime>();
var app = builder.Build();

app.Use(async (context, next) =>
{
    context.Response.Headers.CacheControl = "no-store";
    context.Response.Headers["X-Content-Type-Options"] = "nosniff";
    context.Response.Headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; frame-ancestors 'none'";
    if(context.Request.Path.StartsWithSegments("/api"))
    {
        if(access.Reject(context.Request) is { } rejection)
        {
            context.Response.StatusCode = rejection;
            return;
        }
    }
    try { await next(context); }
    catch(Exception exception) when(exception is ArgumentException or InvalidOperationException or KeyNotFoundException or NotSupportedException)
    {
        context.Response.StatusCode = exception is KeyNotFoundException ? 404 : exception is InvalidOperationException ? 409 : 400;
        await context.Response.WriteAsJsonAsync(new { error = exception.Message });
    }
});
app.UseDefaultFiles();
app.UseStaticFiles();
app.MapGet("/api/status", (DebugRuntime runtime) => runtime.Snapshot());
app.MapGet("/api/batches", (DebugRuntime runtime) => runtime.Batches.Snapshot());
app.MapPost("/api/batches/poll", (DebugRuntime runtime) => runtime.Batches.Snapshot(renew: true));
app.MapPost("/api/start", async (DebugRuntime runtime) => { await runtime.StartAsync(); return Results.Accepted(); });
app.MapPost("/api/stop", async (DebugRuntime runtime) => { await runtime.StopAsync(); return Results.Ok(); });
app.MapGet("/api/cameras/{id}/instructions", (string id, DebugRuntime runtime) => runtime.Session(id).Instructions);
app.MapGet("/api/cameras/{id}/debug", (string id, DebugRuntime runtime) => runtime.Session(id).Snapshot());
app.MapGet("/api/cameras/{id}/registers", (string id, DebugRuntime runtime) => runtime.Session(id).RegisterCatalogSnapshot());
app.MapPost("/api/cameras/{id}/registers/poll", (string id, DebugRuntime runtime) => runtime.Session(id).RegisterCatalogSnapshot(renew: true));
app.MapPost("/api/cameras/{id}/registers/measure", (string id, DebugRuntime runtime) => runtime.Session(id).Registers.Cost.Start());
app.MapGet("/api/cameras/{id}/registers/measure", (string id, DebugRuntime runtime) => runtime.Session(id).Registers.Cost.Snapshot());
app.MapGet("/api/cameras/{id}/watch", (string id, string? viewer, DebugRuntime runtime) => runtime.Session(id).ObservationSnapshot(viewer ?? "main"));
app.MapPut("/api/cameras/{id}/watch", (string id, string? viewer, DebugWatchConfiguration configuration, DebugRuntime runtime) => runtime.Session(id).ConfigureObservation(configuration, viewer ?? "main"));
app.MapPost("/api/cameras/{id}/watch/poll", (string id, string? viewer, DebugWatchAcknowledgement acknowledgement, DebugRuntime runtime) => runtime.Session(id).Observation.Poll(acknowledgement, viewer ?? "main"));
app.MapPost("/api/cameras/{id}/command", (string id, DebugCommand command, DebugRuntime runtime) => { runtime.Session(id).Command(command); return Results.Ok(); });
app.MapPut("/api/cameras/{id}/breakpoints", (string id, DebugBreakpoint[] breakpoints, DebugRuntime runtime) => { runtime.Session(id).SetBreakpoints(breakpoints); return Results.Ok(); });
app.MapPost("/api/cameras/{id}/inspect", async (string id, DebugInspectionRequest request, DebugRuntime runtime, CancellationToken cancellationToken) =>
{
    DebugInspectionResult result = await runtime.Session(id).InspectAsync(request, cancellationToken);
    return Results.Bytes(result.Data, result.ContentType);
});
app.MapGet("/api/cameras/{id}/preview.png", async (string id, DebugRuntime runtime, CancellationToken cancellationToken) =>
    Results.Bytes(await runtime.PreviewAsync(id, cancellationToken), "image/png"));
app.Lifetime.ApplicationStopping.Register(() => app.Services.GetRequiredService<DebugRuntime>().StopAsync().GetAwaiter().GetResult());
await app.RunAsync();
