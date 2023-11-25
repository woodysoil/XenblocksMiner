```
$ mkdir cpp/build && cd cpp/build
$ cmake .. && make
$ ./argon2Verify
$ python3 ../../python/argon2Verify.py
```

output
```
$ ./argon2Verify
Generated Hash: $argon2id$v=19$m=65536,t=1,p=1$JGkeVK+v4kFqglIJfJymdVcnFHU$Nh2ZpZlhq/goSkpboU2k/1v1pjspuNEHhGa18X8HOwfWbF7HQKkBAqtDKgYKM+b6mExoy06O5WADcMhxufdkWQ
Password verified!
```
```
$ python3 ../../python/argon2Verify.py
Generated Hash: $argon2id$v=19$m=65536,t=1,p=1$JGkeVK+v4kFqglIJfJymdVcnFHU$Nh2ZpZlhq/goSkpboU2k/1v1pjspuNEHhGa18X8HOwfWbF7HQKkBAqtDKgYKM+b6mExoy06O5WADcMhxufdkWQ
Invalid password!
```

