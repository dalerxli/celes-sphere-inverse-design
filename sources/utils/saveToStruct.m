function saveToStruct(obj,filename)
varname = inputname(1);
props = properties(obj);
for p=numel(props)
    s.(props{p})=obj.(props{p});
end
eval([varname ' = s']);
save(filename,varname);
end