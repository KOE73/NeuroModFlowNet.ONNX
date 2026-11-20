export function createDebuggerSession(api) {
  function call(id, action, signal, method = 'GET', body, timeout = 15000) {
    return api('/api/cameras/' + encodeURIComponent(id) + '/' + action,{
      method,
      body:body === undefined ? undefined : JSON.stringify(body),
      signal:AbortSignal.any([signal,AbortSignal.timeout(timeout)])
    });
  }

  return {
    async openRegisterVideo(id,selection,signal) {
      return api('/api/cameras/' + encodeURIComponent(id) + '/registers/video', {
        method:'POST', body:JSON.stringify(selection), signal
      });
    },
    async getInstructions(id,signal) { return (await call(id,'instructions',signal)).json(); },
    async sendCommand(id,command,signal) { await call(id,'command',signal,'POST',command); },
    async setBreakpoints(id,breakpoints,signal) { await call(id,'breakpoints',signal,'PUT',breakpoints); },
    async pollRegisterCatalog(id,signal) { return (await call(id,'registers/poll',signal,'POST')).json(); },
    async getObservation(id,signal) { return (await call(id,'watch',signal)).json(); },
    async setObservation(id,configuration,signal) { return (await call(id,'watch',signal,'PUT',configuration)).json(); },
    async pollObservation(id,acknowledgement,signal) { return (await call(id,'watch/poll',signal,'POST',acknowledgement)).json(); },
    async inspect(id,inspection,signal) {
      const response = await call(id,'inspect',signal,'POST',inspection,120000);
      return inspection.format === 'png' ? response.blob() : response.text();
    },
    async getResultImage(id,signal) { return (await call(id,'preview.png',signal)).blob(); }
  };
}
