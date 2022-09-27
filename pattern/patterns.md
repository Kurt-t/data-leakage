# Pattern 1
data = transform(data, suspicious_op(data))
train, test = split(data)
model.fit(train).predict(test)

fixed:
train, test = split(data)
train = transform(train, suspicious_op(train))
test = transform(test, suspicious_op(train))
model.fit(train).predict(test)

# Pattern 2
train, test = split(data) # OR train, test = load(train.csv), load(test.csv)
train = transform(train, suspicious_op(train))
test = transform(test, suspicious_op(test))
model.fit(train).predict(test)

# Pattern 3
data = transform(data, suspicious_op(data))
model.fit(data, split=True)



# Pattern 4
data = transform(data, suspicious_op(data))
train, test = split(data)
model.fit(train).predict(test)

train = transform(train, suspicious_op(train))
train_new, validation = split(train)
model.fit(train_new).predict(validation)



