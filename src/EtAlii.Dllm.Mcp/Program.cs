using EtAlii.Dllm.Core.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

// Model Context Protocol server over stdio. Lets MCP clients (Claude Code, Claude Desktop, IDEs) call the
// deterministic model as a tool. Register it with, for example:
//   claude mcp add dllm -- dotnet run --project src/EtAlii.Dllm.Mcp

var builder = Host.CreateApplicationBuilder(args);

// stdout carries the MCP protocol, so all logging must go to stderr.
builder.Logging.AddConsole(options => options.LogToStandardErrorThreshold = LogLevel.Trace);

builder.Services.AddSingleton(DllmEngine.CreateDefault());
builder.Services
    .AddMcpServer()
    .WithStdioServerTransport()
    .WithToolsFromAssembly();

await builder.Build().RunAsync();
