

import React, { Component } from 'react'
import { Text, View, ToolbarAndroid, StyleSheet, Image, FlatList } from 'react-native'

export default class Test.js extends Component {

    static navigationOptions = { header: null }

    render() {
        return (
            <View style={{ marginTop: 24 }}>


            <ToolbarAndroid 
            style={styles.toolbar} 
            title="Test title" 
            navIcon={require("../assets/images/robot-dev.png")} />



            <Image style={{ width: 50, height: 50 }} source={{ uri: 'https://facebook.github.io/react-native/docs/assets/favicon.png' }}
            />



            <FlatList data={[{ key: 'This is item 1' }, { key: 'This is item 2' }]} renderItem={({ item }) =><Text>{item.key}</Text>}
            />



            <FlatList data={[{ key: 'This is item 1' }, { key: 'This is item 2' }]} renderItem={({ item }) =><Text>{item.key}</Text>}
            />



            <Button
            onPress={onPressLearnMore}
            title="Learn More"
            color="#841584"
            accessibilityLabel="Learn more about this purple button"
            />


            </View>
        )
    }
}

const styles = StyleSheet.create({
    toolbar: {
        backgroundColor: '#2196F3',
        height: 56
    },
});


