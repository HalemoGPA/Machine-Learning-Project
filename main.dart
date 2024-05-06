import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() => runApp(MyApp());
final lightTheme = ThemeData(
  primarySwatch: Colors.blue,
  brightness: Brightness.light,
);
final darkTheme = ThemeData(
  primarySwatch: Colors.blue,
  brightness: Brightness.dark,
);
dynamic currentTheme = isDarkMode ? darkTheme : lightTheme;

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Stroke Classification App',
      theme: currentTheme,
      home: MyHomePage(title: 'Stroke Classification'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  final String title;
  const MyHomePage({Key? key, required this.title}) : super(key: key);

  @override
  _MyHomePageState createState() => _MyHomePageState();
}

bool isDarkMode = true;

class _MyHomePageState extends State<MyHomePage> {
  // bool isDarkMode = false;
  double glucoseLevel = 50;
  double bmi = 20.0; // Default BMI value
  String residenceType = 'Urban';
  String workType = "Private";
  String smokingStatus = "smokes";
  String result = '';
  String gender = "Female";
  int heartDisease = 0;
  int hypertension = 0;
  String married = "Yes";
  double age = 30.0; // Default age
  dynamic apiData = {
    "gender": "Female",
    "age": 72.0,
    "hypertension": 1,
    "heart_disease": 0,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 85.0,
    "bmi": 34.0,
    "smoking_status": "never smoked",
    "smoking_not_found": "False"
  };

  Future<dynamic> post(
      {required String url, @required dynamic body, String? token}) async {
    try {
      print("1");
      // final timeout = const Duration(seconds: 30);
      Map<String, String> header = {"Content-Type": "application/json"};
      if (token != null) {
        header.addAll({'Authorization': 'Bearer $token'});
      }
      print("2");
      http.Response response =
          await http.post(Uri.parse(url), body: body, headers: header);
      print("3");
      if (response.statusCode == 200) {
        print("4");
        return jsonDecode(response.body);
      } else {
        print(
            'there is a problem with status code${response.statusCode} with bode ${jsonDecode(response.body)}');
        // throw Exception(
        // 'there is a problem with status code${response.statusCode} with bode ${jsonDecode(response.body)}');
      }
    } catch (e) {
      print('An error occurred: $e'); // Printing the exception message
    }
  }

  Future<dynamic> classifyStroke({required String uri, String? token}) async {
    Map<String, String> header = {};
    if (token != null) {
      header.addAll({'Authorization': 'Bearer $token'});
    }
    http.Response response = await http.get(Uri.parse(uri), headers: header);
    if (response.statusCode == 200) {
      print("hello");
      return jsonDecode(response.body);
    } else {
      print("hello2");
      throw Exception(
          'there is a problem with status code${response.statusCode}');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
        actions: [
          IconButton(
            icon: Icon(isDarkMode ? Icons.wb_sunny : Icons.nightlight_round),
            onPressed: () {
              setState(() {
                print("hi");

                isDarkMode = !isDarkMode;
                currentTheme = isDarkMode ? darkTheme : lightTheme;
              });
            },
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    'Age:',
                    style: TextStyle(fontSize: 16),
                  ),
                  Container(
                    child: Row(
                      children: [
                        Text(
                          '${age.toStringAsFixed(2)}',
                          style: TextStyle(fontSize: 16),
                        ),
                        Slider(
                          value: age,
                          min: 0,
                          max: 100,
                          divisions: 120,
                          label: age.toStringAsFixed(2).toString(),
                          onChanged: (double value) {
                            setState(() {
                              age = value;
                            });
                          },
                        ),
                      ],
                    ),
                  ),
                ],
              ),
              Container(
                decoration: BoxDecoration(
                  border: Border(
                    bottom: BorderSide(color: Colors.black),
                  ),
                ),
              ),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    'BMI:',
                    style: TextStyle(fontSize: 16),
                  ),
                  Container(
                    child: Row(children: [
                      Text(
                        '${bmi.toStringAsFixed(2)}',
                        style: TextStyle(fontSize: 16),
                      ),
                      Slider(
                        value: bmi,
                        min: 10,
                        max: 50,
                        divisions: 50,
                        label: bmi.toStringAsFixed(2),
                        onChanged: (double value) {
                          setState(() {
                            bmi = value;
                          });
                        },
                      ),
                    ]),
                  ),
                ],
              ),
              Container(
                decoration: BoxDecoration(
                  border: Border(
                    bottom: BorderSide(color: Colors.black),
                  ),
                ),
              ),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    'Glucose Level:',
                    style: TextStyle(fontSize: 16),
                  ),
                  Container(
                    child: Row(
                      children: [
                        Text(
                          '${glucoseLevel.toStringAsFixed(0)}',
                          style: TextStyle(fontSize: 16),
                        ),
                        Slider(
                          value: glucoseLevel,
                          min: 50,
                          max: 300,
                          divisions: 1000,
                          label: glucoseLevel.toStringAsFixed(2),
                          onChanged: (double value) {
                            setState(() {
                              glucoseLevel = value;
                            });
                          },
                        ),
                      ],
                    ),
                  )
                ],
              ),
              Container(
                decoration: BoxDecoration(
                  border: Border(
                    bottom: BorderSide(color: Colors.black),
                  ),
                ),
              ),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text('Gender:'),
                  Container(
                    child: Row(
                      children: [
                        Radio(
                          value: "Male",
                          groupValue: gender,
                          onChanged: (String? value) {
                            setState(() {
                              gender = value!;
                            });
                          },
                        ),
                        Text('Male'),
                        Radio(
                          value: "Female",
                          groupValue: gender,
                          onChanged: (String? value) {
                            setState(() {
                              gender = value!;
                            });
                          },
                        ),
                        Text('Female'),
                      ],
                    ),
                  ),
                ],
              ),
              Container(
                decoration: BoxDecoration(
                  border: Border(
                    bottom: BorderSide(color: Colors.black),
                  ),
                ),
              ),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text('Residence Type'),
                  Container(
                    child: Row(
                      children: [
                        Radio(
                          value: "Urban",
                          groupValue: residenceType,
                          onChanged: (String? value) {
                            setState(() {
                              residenceType = value!;
                            });
                          },
                        ),
                        Text('Urban'),
                        Radio(
                          value: "Rural",
                          groupValue: residenceType,
                          onChanged: (String? value) {
                            setState(() {
                              residenceType = value!;
                            });
                          },
                        ),
                        Text('Rural'),
                      ],
                    ),
                  ),
                ],
              ),
              Container(
                decoration: BoxDecoration(
                  border: Border(
                    bottom: BorderSide(color: Colors.black),
                  ),
                ),
              ),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text('Heart Disease'),
                  Container(
                    child: Row(
                      children: [
                        Radio(
                          value: 0,
                          groupValue: heartDisease,
                          onChanged: (int? value) {
                            setState(() {
                              heartDisease = value!;
                            });
                          },
                        ),
                        Text('0'),
                        Radio(
                          value: 1,
                          groupValue: heartDisease,
                          onChanged: (int? value) {
                            setState(() {
                              heartDisease = value!;
                            });
                          },
                        ),
                        Text('1'),
                      ],
                    ),
                  ),
                ],
              ),
              Container(
                decoration: BoxDecoration(
                  border: Border(
                    bottom: BorderSide(color: Colors.black),
                  ),
                ),
              ),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text('Hypertension'),
                  Container(
                    child: Row(
                      children: [
                        Radio(
                          value: 0,
                          groupValue: hypertension,
                          onChanged: (int? value) {
                            setState(() {
                              hypertension = value!;
                            });
                          },
                        ),
                        Text('0'),
                        Radio(
                          value: 1,
                          groupValue: hypertension,
                          onChanged: (int? value) {
                            setState(() {
                              hypertension = value!;
                            });
                          },
                        ),
                        Text('1'),
                      ],
                    ),
                  ),
                ],
              ),
              Container(
                decoration: BoxDecoration(
                  border: Border(
                    bottom: BorderSide(color: Colors.black),
                  ),
                ),
              ),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text('Ever Married'),
                  Container(
                    child: Row(
                      children: [
                        Radio(
                          value: "No",
                          groupValue: married,
                          onChanged: (String? value) {
                            setState(() {
                              married = value!;
                            });
                          },
                        ),
                        Text('No'),
                        Radio(
                          value: "Yes",
                          groupValue: married,
                          onChanged: (String? value) {
                            setState(() {
                              married = value!;
                            });
                          },
                        ),
                        Text('Yes'),
                      ],
                    ),
                  ),
                ],
              ),
              Container(
                decoration: BoxDecoration(
                  border: Border(
                    bottom: BorderSide(color: Colors.black),
                  ),
                ),
              ),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text("Work Type"),
                  DropdownButton<String>(
                    value: workType,
                    onChanged: (String? newValue) {
                      setState(() {
                        workType = newValue!;
                      });
                    },
                    items: <String>[
                      'Private',
                      'Govt_job',
                      'Self-employed',
                      'Never_worked'
                    ].map<DropdownMenuItem<String>>((String value) {
                      return DropdownMenuItem<String>(
                        value: value,
                        child: Text(value),
                      );
                    }).toList(),
                  ),
                ],
              ),
              Container(
                decoration: BoxDecoration(
                  border: Border(
                    bottom: BorderSide(color: Colors.black),
                  ),
                ),
              ),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text("Smoking Status"),
                  DropdownButton<String>(
                    value: smokingStatus,
                    onChanged: (String? newValue) {
                      setState(() {
                        smokingStatus = newValue!;
                      });
                    },
                    items: <String>['smokes', 'never smoked', 'formerly smoked']
                        .map<DropdownMenuItem<String>>((String value) {
                      return DropdownMenuItem<String>(
                        value: value,
                        child: Text(value),
                      );
                    }).toList(),
                  ),
                ],
              ),
              Container(
                decoration: BoxDecoration(
                  border: Border(
                    bottom: BorderSide(color: Colors.black),
                  ),
                ),
              ),
              SizedBox(height: 20),
              Center(
                child: ElevatedButton(
                  onPressed: () async {
                    print("pressed");
                    apiData["gender"] = gender;
                    apiData["age"] = age;
                    apiData["hypertension"] = hypertension;
                    apiData["heart_disease"] = heartDisease;
                    apiData["ever_married"] = married;
                    apiData["work_type"] = workType;
                    apiData["Residence_type"] = residenceType;
                    apiData["avg_glucose_level"] = glucoseLevel;
                    apiData["bmi"] = bmi;
                    apiData["smoking_status"] = smokingStatus;
                    var response = await post(
                      url: "https://machine-api-eq7w.onrender.com/predict/",
                      body: jsonEncode(apiData),
                    );
                    print(response[
                        'prediction']); // Access the response object here
                    setState(() {
                      result = response['prediction']
                          .toString(); // Update the result
                    });
                  },
                  child: Text('Classify Stroke'),
                ),
              ),
              SizedBox(height: 20),
              Text(
                'Result: ${result == '1' ? "Clear" : "Potential Stroke"}',
                style: TextStyle(fontSize: 16),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
